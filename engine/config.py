"""All runtime configuration in one place.

Every external dependency (URLs, ports, model names) lives here so stages
can be swapped without touching business logic. Override anything via
environment variables (KOKO_*)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    sample_rate: int = 16000
    whisper_chunk_s: float = 1.0        # size of each block fed to ASR (s)
    device: int | None = None           # PortAudio input device; None = system default


@dataclass
class AsrConfig:
    engine: str = "faster_whisper"      # or "whisper_live"
    language: str = "en"
    # faster-whisper (in-process, no server needed)
    model_name: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    # WhisperLive (separate server process)
    whisper_live_url: str = "ws://127.0.0.1:9090"


@dataclass
class LlmConfig:
    base_url: str = "http://xavier:8081/v1"      # llama-server OpenAI-compatible
    model: str = "Qwen3.6-35B-A3B"
    api_key: str = os.getenv("OPENAI_API_KEY_LOCAL", "dummy")
    temperature: float = 0.3
    max_tokens: int = 256


@dataclass
class TtsConfig:
    backend: str = "vieneu"               # "vieneu" | "null"
    vieneu_voice: str = "Adam"              # preset voice name; "" = built-in default
    vieneu_model_path: str = ""         # empty = repo tts_model/ dir
    vieneu_threads: int = 0             # 0 = auto (cores/2, capped at 8)
    output_device: int | None = None    # None = system default output
    sample_rate: int = 48000            # VieNeu v3 Turbo emits 48 kHz


@dataclass
class PhonemizeConfig:
    url: str = "http://127.0.0.1:8788/phonemize"  # sea-g2p wrapper on LLM host


@dataclass
class GateConfig:
    interpretation_delay_s: float = 5.0 # kick off LLM after this much continuous speech
    gap_reset_s: float = 1.2            # silence gap > this means a new 5s cycle begins


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

    def apply_env_overrides(self):
        # source
        self.source.sample_rate = int(os.getenv("KOKO_SAMPLE_RATE", self.source.sample_rate))
        self.asr.language = os.getenv("KOKO_LANGUAGE", self.asr.language)
        self.asr.engine = os.getenv("KOKO_ASR_ENGINE", self.asr.engine)
        self.asr.whisper_live_url = os.getenv("KOKO_WHISPER_LIVE_URL", self.asr.whisper_live_url)
        self.asr.model_name = os.getenv("KOKO_ASR_MODEL", self.asr.model_name)
        self.asr.compute_type = os.getenv("KOKO_ASR_CT", self.asr.compute_type)
        self.llm.base_url = os.getenv("KOKO_LLM_BASE_URL", self.llm.base_url)
        self.llm.model = os.getenv("KOKO_LLM_MODEL", self.llm.model)
        self.llm.temperature = float(os.getenv("KOKO_LLM_TEMP", self.llm.temperature))
        self.gate.interpretation_delay_s = float(os.getenv("KOKO_DELAY_S", self.gate.interpretation_delay_s))
        self.tts.backend = os.getenv("KOKO_TTS", self.tts.backend)
        self.tts.vieneu_model_path = os.getenv("KOKO_VIENEU_PATH", self.tts.vieneu_model_path)
        self.tts.vieneu_voice = os.getenv("KOKO_VIENEU_VOICE", self.tts.vieneu_voice)
        self.tts.vieneu_threads = int(os.getenv("KOKO_VIENEU_THREADS", self.tts.vieneu_threads))
        self.tts.sample_rate = int(os.getenv("KOKO_TTS_SR", self.tts.sample_rate))
        self.phonemize.url = os.getenv("KOKO_PHONEMIZE_URL", self.phonemize.url)


def load_config() -> Config:
    cfg = Config()
    cfg.apply_env_overrides()
    return cfg


SPEAKER_PROMPT = "Không thêm đánh dấu người nói, hoạt động trên màn hình, hay mã markdown. Dịch theo cách tiếp tục câu nói đã dịch từ trước."
