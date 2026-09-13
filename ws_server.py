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
    """Converts inbound PCM16 frames to bus events (mirrors MicSource).

    The client sends ~20 ms frames; whisper needs ~1 s blocks here, so
    frames are packed into whisper_chunk_s-sized chunks before they hit
    the bus. Without the packing, each sliver gets transcribed on its
    own and the VAD filters every one of them to zero text."""

    def __init__(self, cfg, bus: Bus):
        self.bus = bus
        self.block_asr = int(cfg.source.whisper_chunk_s * cfg.source.sample_rate)
        self._packed: list[np.ndarray] = []

    async def feed(self, pcm16_chunk: bytes) -> None:
        x = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self._packed.append(x)
        if self._packed and sum(n.size for n in self._packed) >= self.block_asr:
            chunk = np.concatenate(self._packed)
            self._packed = []
            rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
            db = 20.0 * np.log10(rms + 1e-9)
            active = db > DBFS_FLOOR
            if active:                      # silence chunks hallucinate in whisper
                await self.bus.publish(Event(kind=Kind.SOURCE_CHUNK, payload=chunk))
            await self.bus.publish(Event(
                kind=Kind.SOURCE_SPEAKS,
                payload={"active": active,
                         "dur_s": chunk.size / IN_RATE}))


class WsServer:
    """Holds the long-lived model handles; one pipeline per connection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.busy = asyncio.Lock()
        self.transcriber = None
        self.backend = None

    async def _load_models(self) -> None:
        cfg = self.cfg = load_config()
        log.info("loading models (ASR + TTS)...")
        asr_engines = ("whisper_live", "faster_whisper")
        if cfg.asr.engine not in asr_engines:
            raise ValueError("unknown asr engine %r; expected one of %s"
                             % (cfg.asr.engine, ", ".join(asr_engines)))
        self.transcriber = (WhisperLiveAsr(cfg) if cfg.asr.engine == "whisper_live"
                            else FasterWhisperAsr(cfg))
        tts_backends = ("vieneu", "null")
        if cfg.tts.backend not in tts_backends:
            raise ValueError("unknown tts backend %r; expected one of %s"
                             % (cfg.tts.backend, ", ".join(tts_backends)))
        self.backend = VieneuTts(cfg) if cfg.tts.backend == "vieneu" else NullTts()
        log.info("models ready")

    async def _close_models(self) -> None:
        if self.backend is not None:
            await self.backend.close()
        if self.transcriber is not None:
            await self.transcriber.close()

    async def start(self) -> None:
        await self._load_models()
        try:
            async with websockets.serve(self._wrapped, HOST, PORT, max_size=None):
                log.info("koko tts-server listening on %s:%d", HOST, PORT)
                await asyncio.get_running_loop().create_future()  # serve forever
        finally:
            await self._close_models()

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
        try:
            await ws.send(json.dumps(obj))
        except websockets.ConnectionClosed:
            pass

    async def _session(self, ws) -> None:
        stop = asyncio.Event()
        bus = Bus()
        cfg = self.cfg
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
        transcriber = self.transcriber
        backend = self.backend
        feed = WsFeed(cfg, bus)
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
        for t in tasks:  # stage crashes otherwise vanish unobserved
            t.add_done_callback(
                lambda fut, t=t: (log.error("stage '%s' died: %r", t.get_name(),
                                             fut.exception())
                                  if not fut.cancelled() and fut.exception()
                                  else None))
        out_task = asyncio.create_task(self._drain_out(ws, out_q, stop), name="out")
        text_task = asyncio.create_task(self._forward_texts(bus, ws, stop), name="text")
        try:
            await self._ctl(ws, {"type": "ready"})
            async for msg in ws:
                if isinstance(msg, bytes):
                    if out_q.qsize() < 192:  # downstream slower than capture: drop
                        await feed.feed(msg)
                    continue
                ctl = json.loads(msg)
                if ctl.get("type") == "hello":
                    cfg.asr.language = ctl.get("language", cfg.asr.language)
                    await self._ctl(ws, {"type": "ready"})
                elif ctl.get("type") == "bye":
                    break
                else:
                    await self._ctl(ws, {"type": "error", "detail": "unknown control"})
        finally:
            stop.set()
            all_tasks = [*tasks, out_task, text_task]
            for t in all_tasks:
                t.cancel()
            # gather() can hang on a wedged stage (stuck LLM call, executor
            # task); a hung cleanup would keep `self.busy` locked forever and
            # then reject every later client with "server busy". Bound it.
            await asyncio.wait(all_tasks, timeout=5)
            try:
                await asyncio.wait_for(llm.close(), timeout=3)
            except asyncio.TimeoutError:
                pass

        log.info("client disconnected: %s", ws.remote_address)

    @staticmethod
    async def _forward_texts(bus: Bus, ws, stop: asyncio.Event) -> None:
        """Relay ASR text and LLM turns as websocket control messages."""
        q = bus.subscribe(Kind.PARTIAL_TEXT, Kind.FINAL_TEXT, Kind.SPEAK,
                          Kind.ASSISTANT_CHUNK)
        while not stop.is_set():
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            kind = {"partial_text": "partial", "final_text": "final",
                    "speak": "speak",
                    "assistant_chunk": "translation"}.get(str(ev.kind.value),
                                                          str(ev.kind.value))
            log.info("text[%s]: %r", kind, ev.text)
            await ws.send(json.dumps({
                "type": kind,
                "text": ev.text,
            }))

    @staticmethod
    async def _drain_out(ws, out_q: asyncio.Queue, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                audio, sr = await asyncio.wait_for(out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            pcm = _to_pcm16(audio)
            await ws.send(json.dumps({"type": "audio", "rate": sr}))
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
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    main()
