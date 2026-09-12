"""koko over websockets — hosts the whole interpreter pipeline behind one port.

Listens on 0.0.0.0:6942 (cloudflared-tunnel friendly, one connection at a
time). Protocol on a single socket:

  client -> server:
    binary frames     raw mono PCM16 @ 16 kHz (~20 ms per frame)
    JSON control      {"type": "hello", "language": "en"}
  server -> client:
    {"type":"audio","rate":48} + binary PCM16 chunks (TTS output)
    {"type":"partial","text":...} / {"type":"final","text":...}  live ASR
    {"type":"speak","text":...}              text being interpreted
    {"type":"ready"} / {"type":"error","detail":...}

Run:  python ws_server.py
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
import websockets

from engine.asr import AsrStage, FasterWhisperAsr, WhisperLiveAsr
from engine.bus import Bus
from engine.config import load_config
from engine.events import Event, Kind
from engine.gate import InterpretationGate
from engine.llm import LlmStage
from engine.monitor import Monitor
from engine.tts import NullTts, VieneuTts
from engine.tts import TtsStage

log = logging.getLogger("koko.ws_server")

HOST = "0.0.0.0"
PORT = 6942
IN_RATE = 16_000
DBFS_FLOOR = -45.0


class WsFeed:
    """Converts inbound PCM16 frames to bus events (mirrors MicSource)."""

    def __init__(self, bus: Bus):
        self.bus = bus

    async def feed(self, pcm16_chunk: bytes) -> None:
        x = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-9)
        await self.bus.publish(Event(kind=Kind.SOURCE_CHUNK, payload=x))
        await self.bus.publish(Event(kind=Kind.SOURCE_SPEAKS, payload=bool(db > DBFS_FLOOR)))


class WsServer:
    """Holds the long-lived model handles; one pipeline per connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.busy = asyncio.Lock()

    async def start(self) -> None:
        async with websockets.serve(self._wrapped, HOST, PORT, max_size=None):
            log.info("koko tts-server listening on %s:%d", HOST, PORT)
            await asyncio.get_running_loop().create_future()  # serve forever

    # ---- plumbing per connection ----
    async def _wrapped(self, ws) -> None:
        if self.busy.locked():
            await self._ctl(ws, {"type": "error", "detail": "server busy"})
            return
        async with self.busy:
            try:
                await self._session(ws)
            except Exception:
                log.exception("session crashed")
                try:
                    await self._ctl(ws, {"type": "error", "detail": "server error"})
                except Exception:
                    pass

    async def _ctl(self, ws, obj: dict) -> None:
        await ws.send(json.dumps(obj).encode("utf-8"))

    async def _session(self, ws) -> None:
        stop = asyncio.Event()
        bus = Bus()
        cfg = self.cfg = load_config()
        monitor = Monitor(cfg)

        out_rate = cfg.tts.sample_rate

        class _Sink:
            """AudioPlayer stand-in for TtsStage: wraps an asyncio.Queue."""

            def __init__(self, out_q):
                self.out_q = out_q

            def push(self, audio, sr):
                self.out_q.put_nowait((audio, sr))

            def start(self):
                pass

            def stop(self):
                pass

        out_q: asyncio.Queue = asyncio.Queue(maxsize=256)
        transcriber = (WhisperLiveAsr(cfg) if cfg.asr.engine == "whisper_live"
                       else FasterWhisperAsr(cfg))
        backend = VieneuTts(cfg) if cfg.tts.backend == "vieneu" else NullTts()
        feed = WsFeed(bus)
        gate = InterpretationGate(cfg, bus, monitor)
        llm = LlmStage(cfg, bus, monitor)
        sink = _Sink(out_q)
        tts = TtsStage(cfg, bus, backend, sink)
        tts.sample_rate = out_rate
        tasks = [
            asyncio.create_task(bus.pump(stop), name="bus"),
            asyncio.create_task(gate.run(stop), name="gate"),
            asyncio.create_task(AsrStage(bus, monitor, transcriber).run(stop), name="asr"),
            asyncio.create_task(llm.run(stop, "tiếng Việt"), name="llm"),
            asyncio.create_task(tts.run(stop), name="tts"),
        ]
        out_task = asyncio.create_task(self._drain_out(ws, out_q, stop), name="out")
        text_task = asyncio.create_task(self._forward_texts(bus, ws, stop), name="text")
        await self._ctl(ws, {"type": "ready"})
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    if out_q.qsize() < 192:  # downstream slower than capture: drop
                        await feed.feed(msg)
                    continue
                ctl = json.loads(msg.decode("utf-8"))
                if ctl.get("type") == "hello":
                    cfg.asr.language = ctl.get("language", cfg.asr.language)
                    await self._ctl(ws, {"type": "ready"})
                elif ctl.get("type") == "bye":
                    break
                else:
                    await self._ctl(ws, {"type": "error", "detail": "unknown control"})
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            out_task.cancel()
            text_task.cancel()
            await llm.close()
            await backend.close()

        log.info("client disconnected: %s", ws.remote_address)

    @staticmethod
    async def _forward_texts(bus: Bus, ws, stop: asyncio.Event) -> None:
        """Relay ASR text and LLM turns as websocket control messages."""
        q = bus.subscribe(Kind.PARTIAL_TEXT, Kind.FINAL_TEXT, Kind.SPEAK)
        while not stop.is_set():
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await ws.send(json.dumps({
                "type": {"partial_text": "partial", "final_text": "final",
                         "speak": "speak"}.get(str(ev.kind.value), str(ev.kind.value)),
                "text": ev.text,
            }).encode("utf-8"))

    @staticmethod
    async def _drain_out(ws, out_q: asyncio.Queue, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                audio, sr = await asyncio.wait_for(out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            pcm = _to_pcm16(audio)
            await ws.send(json.dumps({"type": "audio", "rate": sr}).encode("utf-8"))
            await ws.send(pcm)




def _to_pcm16(x: "np.ndarray") -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def main() -> None:
    try:
        asyncio.run(WsServer(load_config()).start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
