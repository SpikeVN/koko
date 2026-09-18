"""All runtime configuration in one place.

Every external dependency (URLs, ports, model names) lives here so stages
can be swapped without touching business logic. Values are loaded once from
a TOML file (`config.toml` by default; `load_config(path)` for per-machine
overrides) — there are no environment variables left. Python 3.11+ uses
stdlib `tomllib`; JetPack 5's Python 3.8 uses the equivalent `tomli` backport.
Unknown keys and wrong value types are rejected loudly instead of silently
falling back (an old typo like `vietneu` vs `vieneu` cost real debugging time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, get_args, get_origin, get_type_hints

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.8 on JetPack 5
    import tomli as tomllib


@dataclass
class SourceConfig:
    sample_rate: int = 16000       # input PCM sample rate (Hz); client sends 16 kHz
    whisper_chunk_s: float = 0.25  # size of each block fed to WhisperLive (s)
    device: Optional[int] = None   # PortAudio input device; None = system default


@dataclass
class AsrConfig:
    language: str = "en"
    auto_detect_language: bool = False  # false = use language; true = Whisper detects per session
    # WhisperLive (separate server process; the only ASR engine)
    whisper_live_url: str = "ws://127.0.0.1:9090"
    whisper_live_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    whisper_live_vad: bool = True                # suppresses non-speech before ASR
    whisper_live_same_output_threshold: int = 5  # balance stability and latency
    # WhisperLive server process (no CLI switches; koko.whisper_live_server reads these)
    whisper_live_host: str = "0.0.0.0"
    whisper_live_port: int = 9090
    whisper_live_backend: str = "faster_whisper"
    whisper_live_max_clients: int = 4
    whisper_live_max_connection_s: int = 600


@dataclass
class LlmConfig:
    base_url: str = "http://xavier:8081/v1"      # llama-server OpenAI-compatible
    model: str = "Qwen3.6-35B-A3B"
    api_key: str = "dummy"          # llama-server ignores it; kept for parity
    temperature: float = 0.3
    max_tokens: int = 256
    context_messages: int = 3       # prior translated sentences included in the system message


@dataclass
class TtsConfig:
    backend: str = "vieneu"               # "vieneu" | "null"
    vieneu_voice: str = "Adam"              # preset voice name; "" = built-in default
    vieneu_model_path: str = ""         # empty = repo tts_model/ dir
    vieneu_voices_path: str = ""        # empty = <vieneu_model_path>/voices_v3_turbo.json
    vieneu_threads: int = 0             # 0 = auto (cores/2, capped at 8)
    output_device: Optional[int] = None  # None = system default output
    sample_rate: int = 48000            # VieNeu v3 Turbo emits 48 kHz
    greedy: bool = False                # temperature 0 (deterministic argmax path)
    execution_provider: str = "auto"    # "auto" | "cuda" | "cpu"


@dataclass
class PhonemizeConfig:
    url: str = "http://127.0.0.1:8788/phonemize"  # sea-g2p wrapper on LLM host


@dataclass
class GateConfig:
    # Space-delimited languages use words; CJK languages use script characters.
    release_words: int = 8   # release a translation burst after this many units
    gap_reset_s: float = 1.2    # silence gap > this, with text buffered, releases early
    no_new_words_s: float = 2.0  # release if Whisper adds no finalized words this long


@dataclass
class Config:
    """Whole-pipeline config"""
    source: SourceConfig = field(default_factory=SourceConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    phonemize: PhonemizeConfig = field(default_factory=PhonemizeConfig)

    @property
    def phonemize_url(self) -> str:
        """Model dir default + phonemize URL live here for stage constructors."""
        return self.phonemize.url


_SECTIONS = ("source", "asr", "gate", "llm", "tts", "phonemize")


def _coerce(name: str, annotation, value):
    """Type-check + normalise one TOML value against a dataclass field type."""
    origin = get_origin(annotation)
    if origin is Union:                             # e.g. Optional[int]
        args = get_args(annotation)
        if value is None:
            if type(None) in args:
                return None
            raise TypeError(f"config {name}: value cannot be null (None)")
        non_null = [a for a in args if a is not type(None)]
        if len(non_null) == 1:
            return _coerce(name, non_null[0], value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"config {name}: expected a boolean, got {value!r}")
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"config {name}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"config {name}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"config {name}: expected a string, got {value!r}")
        return value
    raise TypeError(f"config {name}: unsupported field type {annotation!r}")


def _build_section(section: str, cls, table: dict):
    """Build one config dataclass from its TOML table, rejecting typos/types."""
    hints = get_type_hints(cls)
    for key in table:
        if key not in hints:
            raise ValueError(
                f"unknown config key [{section}].{key} "
                f"(expected one of: {', '.join(sorted(hints))})")
    kwargs = {}
    for name, annotation in hints.items():
        if name in table:
            kwargs[name] = _coerce(f"[{section}].{name}", annotation, table[name])
    return cls(**kwargs)


def load_config(path: Union[str, Path] = "config.toml") -> Config:
    """Load the whole pipeline config from a TOML file.

    The file is required and validated: missing file, unknown sections/keys and
    wrong value types all fail loudly with a clear message.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"config file not found: {p}. Pass --config PATH or run from the repo "
            f"root where ./config.toml lives.")
    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"failed to parse config {p}: {e}") from e

    for key in data:
        if key not in _SECTIONS:
            raise ValueError(
                f"unknown config section [{key}] "
                f"(expected one of: {', '.join(_SECTIONS)})")

    return Config(
        source=_build_section("source", SourceConfig, data.get("source", {})),
        asr=_build_section("asr", AsrConfig, data.get("asr", {})),
        gate=_build_section("gate", GateConfig, data.get("gate", {})),
        llm=_build_section("llm", LlmConfig, data.get("llm", {})),
        tts=_build_section("tts", TtsConfig, data.get("tts", {})),
        phonemize=_build_section("phonemize", PhonemizeConfig, data.get("phonemize", {})),
    )


SPEAKER_PROMPT = "Không thêm đánh dấu người nói, hoạt động trên màn hình, hay mã markdown. Dịch theo cách tiếp tục câu nói đã dịch từ trước."
