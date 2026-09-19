"""Interpretation gate: decides when accumulated source text goes to the LLM."""

from __future__ import annotations

import asyncio
import logging
import time

from koko.engine.bus import Bus
from koko.engine.config import Config
from koko.engine.events import Event, Kind
from koko.engine.monitor import Monitor

log = logging.getLogger("koko.gate")
_CJK_LANGUAGES = {"zh", "ja", "ko"}


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x2E80 <= code <= 0x2FFF
        or 0x3000 <= code <= 0x30FF
        or 0x3400 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
    )


class InterpretationGate:
    def __init__(self, cfg: Config, bus: Bus, monitor: Monitor):
        self.release_words = max(1, cfg.gate.release_words)
        self.gap = cfg.gate.gap_reset_s
        self.no_new_words = max(0.0, cfg.gate.no_new_words_s)
        self.language = cfg.asr.language
        self.bus = bus
        self.monitor = monitor
        self.silence_secs = 0.0
        self.speaking = False
        self.buffer: list[str] = []
        self.buffered_since: float | None = None

    def on_voice(self, active: bool, dur_s: float) -> None:
        if active:
            self.silence_secs = 0.0
        else:
            self.silence_secs += dur_s
        self.speaking = active

    def on_asr_text(self, ev: Event) -> None:
        if ev.text:
            if not self.buffer:
                self.buffered_since = time.monotonic()
            self.buffer.append(ev.text)

    def set_language(self, language: str) -> None:
        if isinstance(language, str) and language:
            self.language = language

    def clear(self) -> None:
        self.buffer.clear()
        self.silence_secs = 0.0
        self.speaking = False
        self.buffered_since = None

    def maybe_fire(self, now: float) -> Event | None:
        if not self.buffer:
            return None
        text = " ".join(self.buffer).strip()
        language = self.language.lower().split("-", 1)[0].split("_", 1)[0]
        if language in _CJK_LANGUAGES:
            units = sum(not ch.isspace() for ch in text)
        else:
            cjk_chars = sum(_is_cjk(ch) for ch in text)
            units = cjk_chars if cjk_chars else len(text.split())
        timed_out = (
            self.no_new_words > 0
            and self.buffered_since is not None
            and now - self.buffered_since >= self.no_new_words
        )
        if self.silence_secs >= self.gap or units >= self.release_words or timed_out:
            self.buffer.clear()
            self.silence_secs = 0.0
            self.buffered_since = None
            return Event(kind=Kind.SPEAK, turn_id=f"u{now:.0f}", text=text)
        return None

    async def run(self, stop: asyncio.Event) -> None:
        voice_q = self.bus.subscribe(Kind.SOURCE_SPEAKS)
        text_q = self.bus.subscribe(Kind.FINAL_TEXT)

        async def voice_loop():
            while not stop.is_set():
                ev = await _get(voice_q, stop)
                if ev is None:
                    break
                p = ev.payload or {}
                self.on_voice(bool(p.get("active")), float(p.get("dur_s", 0.0)))
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

        async def stall_loop():
            while not stop.is_set():
                await asyncio.sleep(0.1)
                fire = self.maybe_fire(time.monotonic())
                if fire is not None:
                    await self.bus.publish(fire)

        await asyncio.gather(voice_loop(), text_loop(), stall_loop())


async def _get(q: asyncio.Queue, stop: asyncio.Event) -> Event | None:
    get_task = asyncio.create_task(q.get())
    stop_task = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait(
            {get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done or stop.is_set():
            return None
        return get_task.result()
    finally:
        for task in (get_task, stop_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(get_task, stop_task, return_exceptions=True)
