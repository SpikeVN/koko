"""Interpretation gate: decides *when* accumulated source text goes to the LLM.

Releases the buffered transcript on EITHER trigger:

* enough words accumulated — once the buffered source reaches
  `gate.release_words` (e.g. 5), a burst is sent to the LLM; or
* a pause — continuous silence longer than `gate.gap_reset_s`
  while anything is buffered ends the utterance and releases early.

Word counting uses `.split()` for space-delimited languages. For Japanese,
Chinese, and Korean, script characters are counted instead because those
languages commonly do not put spaces between words. Buffer holds text from
finals; release gathers whatever is buffered at fire time and clears it.
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


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x2E80 <= code <= 0x2FFF       # CJK radicals and symbols
        or 0x3000 <= code <= 0x30FF   # Japanese kana and punctuation
        or 0x3400 <= code <= 0x9FFF   # CJK ideographs
        or 0xAC00 <= code <= 0xD7AF   # Hangul syllables
        or 0xF900 <= code <= 0xFAFF   # CJK compatibility ideographs
    )


class InterpretationGate:
    def __init__(self, cfg: Config, bus: Bus, monitor: Monitor):
        self.release_words = max(1, cfg.gate.release_words)
        self.gap = cfg.gate.gap_reset_s
        self.bus = bus
        self.monitor = monitor
        self.silence_secs = 0.0    # continuous silence in AUDIO time
        self.speaking = False
        self.buffer: list[str] = []

    def on_voice(self, active: bool, dur_s: float) -> None:
        """Track speech/silence in *audio time*, not wall-clock spacing.

        A voice event arrives once per ASR chunk; measuring wall-clock
        distance between them looked like a fresh pause every chunk, so the
        cycle never accumulated. Accumulating `dur_s` (audio time) instead
        makes a real long pause trigger the early flush.
        """
        if active:
            self.silence_secs = 0.0
        else:
            self.silence_secs += dur_s
        self.speaking = active

    def on_asr_text(self, ev: Event) -> None:
        if ev.text:
            self.buffer.append(ev.text)

    def maybe_fire(self, now: float) -> Event | None:
        """Return a SPEAK event if the buffer should go to the LLM now.

        Fires when the buffered word/script-character count reaches
        `release_words`, or when the speaker has paused (`silence_secs >= gap`)
        with a non-empty buffer. Either way the buffer is cleared, so
        continuous speech keeps producing translation bursts. Returns None
        when there's nothing to say.
        """
        if not self.buffer:
            return None
        text = " ".join(self.buffer).strip()
        cjk_chars = sum(_is_cjk(ch) for ch in text)
        units = cjk_chars if cjk_chars else len(text.split())
        if self.silence_secs >= self.gap or units >= self.release_words:
            self.buffer.clear()
            self.silence_secs = 0.0
            return Event(
                kind=Kind.SPEAK,
                turn_id=f"u{now:.0f}",
                text=text,
            )
        return None

    async def run(self, stop: asyncio.Event) -> None:
        voice_q = self.bus.subscribe(Kind.SOURCE_SPEAKS)
        # NOTE: only FINAL_TEXT drives the release count. Live PARTIAL_TEXT is
        # streamed to the client for instant display but deliberately ignored
        # here, so the LLM is never fed cut-off, in-progress speech.
        text_q = self.bus.subscribe(Kind.FINAL_TEXT)

        async def voice_loop():
            while not stop.is_set():
                ev = await _get(voice_q, stop)
                if ev is None:
                    break
                p = ev.payload or {}
                self.on_voice(bool(p.get("active")),
                              float(p.get("dur_s", 0.0)))
                fire = self.maybe_fire(time.monotonic())
                if fire is not None:
                    await self.bus.publish(fire)

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
