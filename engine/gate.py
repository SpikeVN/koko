"""Interpretation gate: decides *when* accumulated source text goes to the LLM.

Standard-interpretation mode: release the buffered transcript once the
source speaker has been speech-active for `interpretation_delay_s`
(5s) continuously. A pause longer than `gap_reset_s` restarts the 5s
cycle. Buffer holds text from partials *and* finals; release is a
FINAL_TEXT-driven flush (so we hand the LLM confirmed text only).
"""

from __future__ import annotations

import asyncio
import logging
import time

from engine.bus import Bus
from engine.config import Config
from engine.events import Event, Kind
from engine.monitor import Monitor

log = logging.getLogger("koko.gate")


class InterpretationGate:
    def __init__(self, cfg: Config, bus: Bus, monitor: Monitor):
        self.delay = cfg.gate.interpretation_delay_s
        self.gap = cfg.gate.gap_reset_s
        self.bus = bus
        self.monitor = monitor
        self.speech_start: float | None = None
        self.last_voice_ts: float | None = None
        self.fired = False
        self.buffer: list[str] = []

    def on_voice(self, active: bool, now: float) -> None:
        if not active:
            return
        if self.last_voice_ts is not None and now - self.last_voice_ts > self.gap:
            # real pause: start a fresh cycle
            self.speech_start = None
            self.fired = False
        self.last_voice_ts = now
        if self.speech_start is None:
            self.speech_start = now
            self.fired = False

    def on_asr_text(self, ev: Event) -> None:
        if ev.text:
            self.buffer.append(ev.text)

    def maybe_fire(self, now: float) -> Event | None:
        if (
            not self.fired
            and self.buffer
            and self.speech_start is not None
            and now - self.speech_start >= self.delay
        ):
            self.fired = True
            text = " ".join(self.buffer).strip()
            self.buffer.clear()
            return Event(
                kind=Kind.SPEAK,
                turn_id=f"u{self.speech_start:.0f}",
                text=text,
            )
        return None

    async def run(self, stop: asyncio.Event) -> None:
        voice_q = self.bus.subscribe(Kind.SOURCE_SPEAKS)
        text_q = self.bus.subscribe(Kind.PARTIAL_TEXT, Kind.FINAL_TEXT)

        async def voice_loop():
            while not stop.is_set():
                ev = await _get(voice_q, stop)
                if ev is None:
                    break
                self.on_voice(bool(ev.payload), now=time.monotonic())

        async def text_loop():
            while not stop.is_set():
                ev = await _get(text_q, stop)
                if ev is None:
                    break
                self.on_asr_text(ev)
                fire = self.maybe_fire(time.monotonic())
                if fire is not None:
                    await self.bus.publish(fire)

        await asyncio.gather(voice_loop(), text_loop())


async def _get(q: asyncio.Queue, stop: asyncio.Event) -> Event | None:
    get_task = asyncio.create_task(q.get())
    stop_task = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    if stop.is_set():
        get_task.cancel()
        return None
    return get_task.result()
