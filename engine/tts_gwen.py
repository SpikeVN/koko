"""Gwen-TTS voice-cloning backend built on Qwen3-TTS."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from engine.config import Config

log = logging.getLogger("koko.tts_gwen")

_GENERATION_CONFIG = {
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.9,
    "max_new_tokens": 4096,
    "repetition_penalty": 2.0,
    "subtalker_do_sample": True,
    "subtalker_temperature": 0.1,
    "subtalker_top_k": 20,
    "subtalker_top_p": 1.0,
}


class GwenTts:
    """Load Gwen-TTS once and serialize GPU generation calls."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self.sample_rate = cfg.tts.sample_rate
        self._ref_audio, self._ref_text = self._reference()

    def _reference(self) -> tuple[str, str]:
        tts = self._cfg.tts
        if tts.gwen_ref_audio:
            if not tts.gwen_ref_text:
                raise ValueError("[tts].gwen_ref_text is required with gwen_ref_audio")
            return tts.gwen_ref_audio, tts.gwen_ref_text
        if not tts.gwen_data_path:
            raise ValueError(
                "set [tts].gwen_data_path for gwen_speaker, or set "
                "gwen_ref_audio and gwen_ref_text")

        data_dir = Path(tts.gwen_data_path)
        try:
            speakers = json.loads((data_dir / "ref_info.json").read_text(encoding="utf-8"))
            speaker = speakers[tts.gwen_speaker]
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Gwen reference metadata not found: {data_dir / 'ref_info.json'}") from exc
        except KeyError as exc:
            raise ValueError(
                f"unknown Gwen speaker {tts.gwen_speaker!r}; "
                f"available: {', '.join(speakers)}") from exc
        ref_audio = data_dir / speaker["audio_path"]
        if not ref_audio.is_file():
            # Gwen's metadata paths are relative to the model root, while
            # gwen_data_path deliberately points at its data/ directory.
            ref_audio = data_dir.parent / speaker["audio_path"]
        if not ref_audio.is_file():
            raise FileNotFoundError(f"Gwen reference audio not found: {ref_audio}")
        return str(ref_audio), speaker["text"]

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Gwen TTS requires the optional 'gwen' dependency group. "
                "Run: uv sync --group gwen") from exc

        try:
            dtype = getattr(torch, self._cfg.tts.gwen_dtype)
        except AttributeError as exc:
            raise ValueError(
                f"unknown Gwen dtype {self._cfg.tts.gwen_dtype!r}") from exc
        attention = self._cfg.tts.gwen_attention
        if attention == "auto":
            try:
                import flash_attn  # noqa: F401
                attention = "flash_attention_2"
            except Exception:
                attention = "sdpa"
        self._model = Qwen3TTSModel.from_pretrained(
            self._cfg.tts.gwen_model_path,
            device_map=self._cfg.tts.gwen_device,
            dtype=dtype,
            attn_implementation=attention,
        )
        log.info("Gwen TTS loaded from %s", self._cfg.tts.gwen_model_path)

    def synth(self, text: str) -> np.ndarray:
        with self._lock:
            self._load()
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=self._cfg.tts.gwen_language,
                ref_audio=self._ref_audio,
                ref_text=self._ref_text,
                **_GENERATION_CONFIG,
            )
        wav = wavs[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        self.sample_rate = int(sample_rate)
        return np.asarray(wav, dtype=np.float32).reshape(-1)
