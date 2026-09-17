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

Run:  uv run koko-server [--config ./config.toml]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import numpy as np
import websockets

from engine.asr import AsrStage, WhisperLiveAsr
from engine.bus import Bus
from engine.config import load_config
from engine.events import Event, Kind
from engine.gate import InterpretationGate
from engine.llm import LlmStage
from engine.monitor import Monitor
from engine.tts import GwenTts, NullTts, VieneuTts
from engine.tts import TtsStage

log = logging.getLogger("koko.ws_server")

HOST = "0.0.0.0"
PORT = 6942
IN_RATE = 16_000
DBFS_FLOOR = -45.0


class WsFeed:
    """Converts inbound PCM16 frames to bus events (mirrors MicSource).

    The client sends ~20 ms frames; they are packed into 250 ms blocks for
    WhisperLive. This matches its reference client (4,096 samples) while
    avoiding a websocket call per tiny capture frame."""

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

    def __init__(self, cfg, event_sink=None):
        self.cfg = cfg
        self.busy = asyncio.Lock()
        self.transcriber = None
        self.backend = None
        self.tts_voices: list[tuple[str, str]] = []
        # optional UI hook: called with each (kind, text) as the pipeline
        # produces them, so a textual frontend can render live state.
        self.event_sink = event_sink
        # live registry of connected clients: websocket -> status dict. The
        # TUI reads this to let the operator pick a client and watch it.
        self.clients: dict = {}

    async def _load_models(self) -> None:
        cfg = self.cfg                      # already loaded from config.toml
        log.info("loading models (ASR + TTS)...")
        # WhisperLive is the only ASR engine; TTS backend is still validated
        # so a config typo fails loudly instead of silently falling back.
        self.transcriber = WhisperLiveAsr(cfg)
        await self.transcriber.warmup()
        tts_backends = ("vieneu", "gwen", "null")
        if cfg.tts.backend not in tts_backends:
            raise ValueError("unknown tts backend %r; expected one of %s"
                             % (cfg.tts.backend, ", ".join(tts_backends)))
        if cfg.tts.backend == "vieneu":
            self.backend = VieneuTts(cfg)
            self.tts_voices = self.backend.list_voices()
        elif cfg.tts.backend == "gwen":
            self.backend = GwenTts(cfg)
        else:
            self.backend = NullTts()
        log.info("models ready")

    async def _close_models(self) -> None:
        if self.backend is not None:
            await self.backend.close()
        if self.transcriber is not None:
            await self.transcriber.close()

    async def serve(self) -> None:
        """Load models, serve websocket connections until cancelled."""
        await self._load_models()
        server = await websockets.serve(self._wrapped, HOST, PORT, max_size=None)
        log.info("koko tts-server listening on %s:%d", HOST, PORT)
        try:
            await asyncio.get_running_loop().create_future()  # serve forever
        finally:
            server.close()
            await server.wait_closed()
            await self._close_models()

    def start(self) -> None:
        """Blocking entry point for the headless server."""
        try:
            asyncio.run(self.serve())
        except KeyboardInterrupt:
            pass

    # ---- plumbing per connection ----
    async def _wrapped(self, ws) -> None:
        if self.busy.locked():
            await self._ctl(ws, {"type": "error", "detail": "server busy"})
            return
        self._register_client(ws)
        try:
            async with self.busy:
                try:
                    await self._session(ws)
                except Exception:
                    log.exception("session crashed")
                    try:
                        await self._ctl(ws, {"type": "error", "detail": "server error"})
                    except Exception:
                        pass
        finally:
            self.clients.pop(ws, None)

    def _register_client(self, ws) -> None:
        self.clients[ws] = {
            "addr": str(ws.remote_address),
            "connected_at": time.monotonic(),
            "language": self.cfg.asr.language,
            "bytes_rx": 0,
            "frames_rx": 0,
            "speaking": False,
            "last": "connected",
            "last_at": time.monotonic(),
        }

    def _bump_client(self, ws, ev: Event) -> None:
        """Refresh a client's live status from a bus event."""
        info = self.clients.get(ws)
        if info is None:
            return
        info["last_at"] = time.monotonic()
        k, text = ev.kind, ev.text
        if k is Kind.SOURCE_SPEAKS:
            info["speaking"] = bool(
                ev.payload.get("active")) if isinstance(ev.payload, dict) else False
        elif k is Kind.PARTIAL_TEXT and text:
            info["last"] = f"hearing: {text}"
        elif k is Kind.FINAL_TEXT and text:
            info["last"] = f"heard: {text}"
        elif k is Kind.SPEAK and text:
            info["last"] = f"translating: {text}"
        elif k is Kind.ASSISTANT_CHUNK and text:
            info["last"] = f"translated: {text}"

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
        ui_task = (asyncio.create_task(self._forward_ui(bus, stop, ws), name="ui-fwd")
                   if self.event_sink is not None else None)
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    info = self.clients.get(ws)
                    if info is not None:
                        info["bytes_rx"] += len(msg)
                        info["frames_rx"] += 1
                    if out_q.qsize() < 192:  # downstream slower than capture: drop
                        await feed.feed(msg)
                    continue
                ctl = json.loads(msg)
                if ctl.get("type") == "hello":
                    cfg.asr.language = ctl.get("language", cfg.asr.language)
                    gate.set_language(cfg.asr.language)
                    auto_detect = ctl.get("asr_auto_detect", cfg.asr.auto_detect_language)
                    if isinstance(auto_detect, bool):
                        cfg.asr.auto_detect_language = auto_detect
                    info = self.clients.get(ws)
                    if info is not None:
                        info["language"] = cfg.asr.language
                    await self._ctl(ws, {
                        "type": "ready",
                        "tts_voices": self.tts_voices,
                        "tts_voice": getattr(self.backend, "_voice", None),
                        "asr_language": cfg.asr.language,
                        "asr_auto_detect": cfg.asr.auto_detect_language,
                    })
                elif ctl.get("type") == "bye":
                    break
                elif ctl.get("type") == "clear_context":
                    gate.clear()
                    llm.clear_context()
                    await self._ctl(ws, {"type": "clear_context"})
                elif ctl.get("type") == "tts_voice":
                    voice = ctl.get("voice")
                    if not isinstance(voice, str):
                        await self._ctl(ws, {"type": "error", "detail": "tts_voice requires a string voice"})
                    elif not isinstance(backend, VieneuTts):
                        await self._ctl(ws, {"type": "error", "detail": "voice selection requires the vieneu backend"})
                    else:
                        try:
                            backend.set_voice(voice)
                        except ValueError as exc:
                            await self._ctl(ws, {"type": "error", "detail": str(exc)})
                        else:
                            await self._ctl(ws, {"type": "tts_voice", "voice": voice})
                elif ctl.get("type") == "asr_language":
                    language = ctl.get("language")
                    if not isinstance(language, str) or not language:
                        await self._ctl(ws, {"type": "error", "detail": "asr_language requires a language code"})
                    else:
                        cfg.asr.language = language
                        cfg.asr.auto_detect_language = False
                        gate.set_language(language)
                        await self._ctl(ws, {
                            "type": "asr_language", "language": language,
                            "asr_auto_detect": False,
                        })
                elif ctl.get("type") == "asr_auto_detect":
                    enabled = ctl.get("enabled")
                    if not isinstance(enabled, bool):
                        await self._ctl(ws, {"type": "error", "detail": "asr_auto_detect requires a boolean enabled"})
                    else:
                        cfg.asr.auto_detect_language = enabled
                        await self._ctl(ws, {"type": "asr_auto_detect", "enabled": enabled})
                else:
                    await self._ctl(ws, {"type": "error", "detail": "unknown control"})
        finally:
            stop.set()
            all_tasks = [*tasks, out_task, text_task]
            if ui_task is not None:
                all_tasks.append(ui_task)
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

    async def _forward_ui(self, bus: Bus, stop: asyncio.Event, ws) -> None:
        """Update the client tracker and feed events to the event_sink."""
        q = bus.subscribe(Kind.PARTIAL_TEXT, Kind.FINAL_TEXT, Kind.SPEAK,
                          Kind.ASSISTANT_CHUNK, Kind.SOURCE_SPEAKS)
        while not stop.is_set():
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            self._bump_client(ws, ev)
            if self.event_sink is not None:
                self.event_sink(ev)

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
    ap = argparse.ArgumentParser(description="koko websocket interpreter server")
    ap.add_argument("--config", default="config.toml",
                    help="path to a config.toml (default: ./config.toml); "
                         "use a per-machine file to point at different URLs")
    args = ap.parse_args()

    WsServer(load_config(args.config)).start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
