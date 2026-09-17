"""ASR stage: streaming transcription.

One engine behind a small interface:

* WhisperLiveAsr -- connects to a running WhisperLive legacy websocket
  server (finance: live, low latency, requires a GPU server started
  separately; see README / whisper-live.sh). It is the only ASR path —
  the former in-process faster-whisper backend was removed.

Publishes FINAL_TEXT for confirmed output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import numpy as np

from engine.bus import Bus
from engine.config import Config
from engine.events import Event, Kind
from engine.monitor import Monitor

log = logging.getLogger("koko.asr")


class Transcriber:
    """Strategy interface for streaming ASR transports."""

    async def transcribe(self, chunk: np.ndarray) -> tuple[list[str], str, str]:
        raise NotImplementedError

    async def send(self, chunk: np.ndarray) -> None:
        raise NotImplementedError

    async def drain(self) -> tuple[list[str], str]:
        raise NotImplementedError

    async def close(self) -> None:  # optional
        return None


class WhisperLiveAsr(Transcriber):
    """Transcriber that speaks the whisper-live 0.10 websocket protocol.

    whisper-live expects a JSON config handshake as the first message on a
    connection, then pushes updates asynchronously: an initial
    SERVER_READY burst followed by rolling segment lists (the last N
    segments are re-sent on every message, so dedupe is mandatory). We keep
    one persistent socket — koko handles one session at a time (`busy`
    lock) — and lazily reconnect + re-handshake when the socket dies or a
    session `hello` changes the language.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ws = None
        self._handshake_lang: tuple[bool, str] | None = None
        self._seen: set[tuple] = set()

    async def _connect(self) -> None:
        """Open a fresh socket and complete the whisper-live handshake."""
        import websockets

        cfg = self.cfg.asr
        self._ws = await websockets.connect(cfg.whisper_live_url)
        await self._ws.send(json.dumps({
            "uid": str(uuid.uuid4()),
            "language": None if cfg.auto_detect_language else cfg.language,
            "task": "transcribe",
            "model": cfg.whisper_live_model,
            "use_vad": cfg.whisper_live_vad,
            "same_output_threshold": cfg.whisper_live_same_output_threshold,
            "send_last_n_segments": 10,
            "audio_format": "int16",               # matches our PCM16 wire
        }))
        try:
            first = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=10))
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"whisper-live {cfg.whisper_live_url} never answered the "
                f"handshake (timeout)") from None
        if first.get("message") == "SERVER_READY":
            return
        status = first.get("status")
        if status in ("WAIT", "ERROR"):
            await self._ws.close()
            self._ws = None
            raise ConnectionError(
                f"whisper-live handshake refused ({status}): {first}")
        raise RuntimeError(f"whisper-live unexpected handshake reply: {first!r}")

    async def warmup(self) -> None:
        """Load the shared WhisperLive model before client audio starts."""
        if self._ws is not None:
            return
        await self._connect()
        self._handshake_lang = (
            self.cfg.asr.auto_detect_language, self.cfg.asr.language)
        self._seen.clear()

    async def transcribe(self, chunk: np.ndarray) -> tuple[list[str], str, str]:
        """Backwards-compatible one-shot: send a chunk then drain."""
        await self.send(chunk)
        new_texts, partial = await self.drain()
        return new_texts, partial, self.cfg.asr.language

    async def send(self, chunk: np.ndarray) -> None:
        """Ensure a live connection, then stream `chunk` (mono float in -1..1)
        to whisper-live as PCM16."""
        cfg = self.cfg.asr
        try:
            language_mode = (cfg.auto_detect_language, cfg.language)
            lang_changed = (self._handshake_lang is not None
                             and language_mode != self._handshake_lang)
            if self._ws is None or lang_changed:
                if self._ws is not None:           # re-handshake: drop old socket
                    await self._ws.close()
                    self._ws = None
                await self._connect()
                self._handshake_lang = language_mode
                self._seen.clear()                 # fresh connection, fresh dedupe
            pcm16 = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
            assert self._ws is not None
            await self._ws.send(pcm16)
        except Exception as exc:
            log.warning("whisper-live connection lost (%s), reconnecting", exc)
            self._ws = None

    async def drain(self) -> tuple[list[str], str]:
        """Collect segments the server pushed since the last drain.

        Called continuously (independent of sends) so live in-progress
        partials reach the client as fast as whisper produces them. Returns
        (newly-completed texts, latest in-progress partial). No-op before a
        connection exists (no chunk has been sent yet).
        """
        if self._ws is None:
            return [], ""
        try:
            return await self._drain()
        except Exception as exc:
            # A server restart or lifetime close can happen while idle. The
            # next audio chunk reconnects; never take the ASR stage down.
            log.warning("whisper-live connection lost (%s), reconnecting", exc)
            self._ws = None
            return [], ""

    async def _drain(self) -> tuple[list[str], str]:
        """Collect segments queued since the last drain.

        whisper-live re-sends the last N segments on every message, so only
        unseen (start, end, text) keys with `completed: True` count as new
        finals.  The trailing in-progress segment (`completed: False`) is
        returned separately as the live partial transcript so the client can
        display it as fast as whisper produces it.
        """
        assert self._ws is not None
        new_texts: list[str] = []
        partial = ""
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=0.05))
            except asyncio.TimeoutError:
                break
            for seg in msg.get("segments", []):
                text = (seg.get("text") or "").strip()
                if seg.get("completed"):
                    key = (seg.get("start"), seg.get("end"), seg.get("text"))
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                    if text:
                        new_texts.append(text)
                elif text:
                    partial = text  # in-progress segment = live working text
        return new_texts, partial

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class AsrStage:
    """Consumes SOURCE_CHUNK events, emits text events onto the bus."""

    def __init__(self, bus: Bus, monitor: Monitor, transcriber: Transcriber):
        self.bus = bus
        self.monitor = monitor
        self.transcriber = transcriber
        self._last_partial = ""

    async def run(self, stop: asyncio.Event) -> None:
        q = self.bus.subscribe(Kind.SOURCE_CHUNK)
        # Send audio as chunks arrive, but keep draining on a short timer even
        # between sends so whisper-live's live partials reach the client at
        # whisper speed (not once per 250 ms input chunk).
        while not stop.is_set():
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.05)
            except asyncio.TimeoutError:
                ev = None
            if ev is not None:
                await self.transcriber.send(ev.payload)
            new_texts, partial = await self.transcriber.drain()
            for text in new_texts:
                if text:
                    await self.bus.publish(Event(kind=Kind.FINAL_TEXT, text=text))
                    self.monitor.observe(Event(kind=Kind.FINAL_TEXT))
                    self._last_partial = ""  # segment completed; drop stale partial
            if partial and partial != self._last_partial:
                self._last_partial = partial
                await self.bus.publish(Event(kind=Kind.PARTIAL_TEXT, text=partial))
                self.monitor.observe(Event(kind=Kind.PARTIAL_TEXT))
