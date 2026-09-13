"""Interpretation gate: decides *when* accumulated source text goes to the LLM.

Standard-interpretation mode: release the buffered transcript once the
source speaker has accumulated `interpretation_delay_s` of speech
(measured in AUDIO time, not event spacing). Continuous silence longer
than `gap_reset_s` restarts the cycle. Buffer holds text from partials
*and* finals; release gathers whatever is buffered at fire time.
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
        self.speech_secs = 0.0     # speech duration in AUDIO time this cycle
        self.silence_secs = 0.0    # continuous silence in AUDIO time
        self.speaking = False
        self.flush = False         # buffer release pending at a pause boundary
        self.buffer: list[str] = []

    def on_voice(self, active: bool, dur_s: float) -> None:
        """Track speech/silence in *audio time*, not wall-clock spacing.

        Measuring the distance between voice events was wrong in the
        websocket path: a voice event only arrives once per ~1 s chunk
        and ASR jitter stretched the spacing past gap_reset_s, so every
        chunk looked like a fresh pause and the cycle never accumulated.
        """
        if active:
            self.silence_secs = 0.0
            self.speech_secs += dur_s
            if not self.speaking:
                log.debug("speech started (%.2fs audio so far)",
                          self.speech_secs)
        else:
            self.silence_secs += dur_s
            if self.silence_secs >= self.gap:
                if self.speech_secs > 0:
                    log.debug("silence %.2fs: new cycle", self.silence_secs)
                self.speech_secs = 0.0
                self.flush = True   # utterance ended: release the buffer
        self.speaking = active

    def on_asr_text(self, ev: Event) -> None:
        if ev.text:
            self.buffer.append(ev.text)

    def maybe_fire(self, now: float) -> Event | None:
        # Fire when the burst of speech reaches interpretation_delay_s, or
        # when a pause of gap_reset_s ends an utterance. Both counters reset
        # after each fire, so continuous speech keeps producing translations.
        if self.buffer and (self.speech_secs >= self.delay or self.flush):
            self.flush = False
            text = " ".join(self.buffer).strip()
            self.buffer.clear()
            self.speech_secs = 0.0   # restart the cycle for continuous speech
            return Event(
                kind=Kind.SPEAK,
                turn_id=f"u{now:.0f}",
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
                p = ev.payload or {}
                self.on_voice(bool(p.get("active")),
                              float(p.get("dur_s", 0.0)))

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
