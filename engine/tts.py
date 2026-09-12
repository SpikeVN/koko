"""TTS stage + audio player.

TtsBackend: anything that turns text -> np.ndarray[float32] audio.
  * VieneuTts  -- local vieneu-tts model (pointed at its import path / weights)
  * NullTts    -- drops audio (offline testing / speech-free dry runs)

Playback lives in `AudioPlayer`, a queue of sentence-waveforms drained
by a sounddevice output stream. Live-sync TTS behaves well as long as
its synth is faster than real time; the queue exposes how far behind
it's falling.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from engine.bus import Bus
from engine.config import Config
from engine.events import Event, Kind

log = logging.getLogger("koko.tts")


class TtsBackend:
    async def speak(self, text: str) -> np.ndarray:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class NullTts(TtsBackend):
    """Logs text, emits nothing audibly (dry-run mode)."""

    async def speak(self, text: str) -> np.ndarray:
        await asyncio.sleep(0.05)
        return np.zeros(0, dtype=np.float32)


class VieneuTts(TtsBackend):
    """VieNeu-TTS v3 Turbo via the local torch-free ONNX pipeline
    (engine/tts_vieneu.py), phonemized remotely (engine/phonemize.py).

    Heavy synth runs in a worker thread so the asyncio loop never blocks."""

    def __init__(self, cfg: Config):
        from engine.phonemize import Phonemizer
        self._cfg = cfg
        self.sample_rate = cfg.tts.sample_rate
        self._engine = None
        self._lock = threading.Lock()
        self._phonemizer = Phonemizer(cfg)
        self._model_path = cfg.tts.vieneu_model_path or str(
            Path(__file__).resolve().parent.parent / "tts_model")
        self._voice = cfg.tts.vieneu_voice

    def _load(self) -> None:
        if self._engine is not None:
            return
        from engine.tts_vieneu import VieneuLite

        self._engine = VieneuLite(
            model_dir=self._model_path,
            voice=self._voice,
            threads=self._cfg.tts.vieneu_threads,
        )
        self.sample_rate = VieneuLite.SAMPLE_RATE
        log.info("vieneu (ONNX) loaded from %s (voice=%s)", self._model_path, self._voice)

    async def speak(self, text: str) -> np.ndarray:
        phonemes = await self._phonemizer.run(text)
        if not phonemes:
            return np.zeros(0, dtype=np.float32)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synth, phonemes)

    def _synth(self, phonemes: str) -> np.ndarray:
        with self._lock:
            self._load()
        wav = self._engine.synth(phonemes)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return np.asarray(wav, dtype=np.float32)

    async def close(self) -> None:
        await self._phonemizer.close()


class TtsStage:
    def __init__(self, cfg: Config, bus: Bus, backend: TtsBackend, player: "AudioPlayer"):
        self.bus = bus
        self.backend = backend
        self.player = player
        self.sample_rate = cfg.tts.sample_rate if cfg.tts.backend != "null" else 16000

    async def run(self, stop: asyncio.Event) -> None:
        q = self.bus.subscribe(Kind.ASSISTANT_CHUNK)
        while True:
            ev = await _get(q, stop)
            if ev is None:
                break
            audio = await self.backend.speak(ev.text)
            if audio.size:
                self.player.push(audio, self.sample_rate)
                log.info("queued %d chars of speech (%.2fs)", len(ev.text), len(audio) / self.sample_rate)


class AudioPlayer:
    """Thread-safe FIFO of waveforms played through sounddevice."""

    def __init__(self, output_device: int | None = None):
        self.q: "queue.SimpleQueue[tuple[np.ndarray, int]]" = queue.SimpleQueue()
        self._output_device = output_device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="koko-player", daemon=True)
            self._thread.start()

    def _run(self):
        current_stream: sd.OutputStream | None = None
        while not self._stop.is_set():
            try:
                audio, sr = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if current_stream is None or current_stream.samplerate != sr:
                if current_stream is not None:
                    current_stream.stop()
                    current_stream.close()
                current_stream = sd.OutputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    device=self._output_device,
                )
                current_stream.start()
            current_stream.write(audio)
        if current_stream is not None:
            current_stream.stop()
            current_stream.close()

    def stop(self):
        self._stop.set()


# helper to read from an asyncio Queue with graceful shutdown


async def _get(q: asyncio.Queue, stop: asyncio.Event):
    get_task = asyncio.create_task(q.get())
    stop_task = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    if stop.is_set():
        get_task.cancel()
        return None
    return get_task.result()
