"""Wires all stages to the bus and runs them to completion.

Stage graph:
  mic source (thread) -> SOURCE_CHUNK / SOURCE_SPEAKS
  ASR stage           -> FINAL_TEXT
  gate                -> SPEAK (once per 5s speech cycle)
  LLM stage           -> ASSISTANT_CHUNK (sentence-sized)
  TTS stage -> player (audio out)

Ordering guarantee: the bus fans each event out to per-subscriber
queues, so consumers can't reorder, only lag behind; queue depth is
the backpressure signal.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from engine.bus import Bus
from engine.config import Config
from engine.monitor import Monitor
from engine.source import MicSource
from engine.asr import AsrStage, FasterWhisperAsr, WhisperLiveAsr
from engine.gate import InterpretationGate
from engine.llm import LlmStage
from engine.tts import AudioPlayer, NullTts, TtsStage, VieneuTts

log = logging.getLogger("koko.pipeline")


def build_transcriber_and_backend(cfg: Config):
    if cfg.asr.engine == "whisper_live":
        transcriber = WhisperLiveAsr(cfg)
    else:
        transcriber = FasterWhisperAsr(cfg)

    if cfg.tts.backend == "vieneu":
        backend = VieneuTts(cfg)
    else:
        backend = NullTts()
    return transcriber, backend


async def run_pipeline(cfg: Config, stop: asyncio.Event) -> None:
    bus = Bus()
    monitor = Monitor(cfg)
    tasks = [
        asyncio.create_task(bus.pump(stop), name="bus"),
        asyncio.create_task(monitor.run_report_loop(stop), name="monitor"),
    ]

    transcriber, backend = build_transcriber_and_backend(cfg)
    player = AudioPlayer(cfg.tts.output_device)

    # mic in its own thread (sounddevice callback); publishes via bus.put()
    stop_thread = threading.Event()
    mic = MicSource(cfg, bus, monitor)
    mic_thread = threading.Thread(target=mic.run_forever, args=(stop_thread,), daemon=True)
    mic_thread.start()
    player.start()

    stages = [
        ("asr", asr_stage := AsrStage(bus, monitor, transcriber)),
        ("gate", gate := InterpretationGate(cfg, bus, monitor)),
        ("llm", llm := LlmStage(cfg, bus, monitor)),
        ("tts", tts := TtsStage(cfg, bus, backend, player)),
    ]
    for name, stage in stages:
        tasks.append(asyncio.create_task(stage.run(stop), name=name))

    try:
        await asyncio.gather(*tasks)
    finally:
        stop_thread.set()
        mic_thread.join(timeout=5)
        await llm.close()
        await backend.close()
        player.stop()
        for t in tasks:
            t.cancel()


def main() -> None:
    from engine.config import load_config

    cfg = load_config()

    async def _runner():
        stop = asyncio.Event()
        task = asyncio.create_task(run_pipeline(cfg, stop))
        try:
            await task
        except asyncio.CancelledError:
            pass

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        log.info("shutting down (ctrl-c)")


if __name__ == "__main__":
    main()
