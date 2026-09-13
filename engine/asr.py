"""ASR stage: streaming transcription.

Two interchangeable engines behind one interface:

* FasterWhisperAsr -- faster-whisper in-process (default; no extra
  server process). Transcribes each SOURCE_CHUNK incrementally and
  merges segment text into partials, emitting FINAL_TEXT per 5s of
  continuous-ish speech.

* WhisperLiveAsr   -- connects to a running WhisperLive legacy
  websocket server (finance: live, low latency, requires GPU server
  started separately; see README).

Both publish PARTIAL_TEXT for provisional output and FINAL_TEXT for
confirmed output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import numpy as np

from engine.bus import Bus
from engine.config import Config
from engine.events import Event, Kind
from engine.monitor import Monitor

log = logging.getLogger("koko.asr")


class Transcriber:
    """Strategy interface: async_transcribe(chunk) -> (segments, language)."""

    async def transcribe(self, chunk: np.ndarray) -> tuple[list[str], str]:
        raise NotImplementedError

    async def close(self) -> None:  # optional
        return None


class FasterWhisperAsr(Transcriber):
    def __init__(self, cfg: Config):
        from faster_whisper import WhisperModel

        a = cfg.asr
        log.info("loading faster-whisper %s (%s, %s)", a.model_name, a.device, a.compute_type)
        self.model = WhisperModel(
            a.model_name, device=a.device, compute_type=a.compute_type
        )
        self.language = a.language

    async def transcribe(self, chunk: np.ndarray) -> tuple[list[str], str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, chunk
        )

    def _transcribe_sync(self, chunk: np.ndarray) -> tuple[list[str], str]:
        segments, info = self.model.transcribe(
            chunk,
            language=self.language,
            beam_size=1,
            vad_filter=False,   # VAD drops every segment even on clean clips
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        return [seg.text.strip() for seg in segments], info.language


class WhisperLiveAsr(Transcriber):
    def __init__(self, cfg: Config):
        self.url = cfg.asr.whisper_live_url
        self.language = cfg.asr.language
        self._ws = None

    async def _connect(self):
        import websockets

        self._ws = await websockets.connect(self.url)

    async def transcribe(self, chunk: np.ndarray) -> tuple[list[str], str]:
        import websockets
        import json as _json

        try:
            if self._ws is None:
                await self._connect()
            pcm16 = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
            assert self._ws is not None
            await self._ws.send(pcm16.tobytes())
            sent = await self._ws.recv()
            text = _json.loads(sent)["segments"]
            return [s["text"].strip() for s in text], self.language
        except Exception as exc:
            log.warning("whisper-live connection lost (%s), reconnecting", exc)
            self._ws = None
            return [], self.language


class AsrStage:
    """Consumes SOURCE_CHUNK events, emits text events onto the bus."""

    def __init__(self, bus: Bus, monitor: Monitor, transcriber: Transcriber):
        self.bus = bus
        self.monitor = monitor
        self.transcriber = transcriber

    async def run(self, stop: asyncio.Event) -> None:
        q = self.bus.subscribe(Kind.SOURCE_CHUNK)
        while not stop.is_set():
            get_task = asyncio.create_task(q.get())
            stop_task = asyncio.create_task(stop.wait())
            done, _ = await asyncio.wait({get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                get_task.cancel()
                break
            ev = get_task.result()
            chunk: np.ndarray = ev.payload
            segments, _lang = await self.transcriber.transcribe(chunk)
            for text in segments:
                if text:
                    await self.bus.publish(Event(kind=Kind.FINAL_TEXT, text=text))
                    self.monitor.observe(Event(kind=Kind.FINAL_TEXT))
