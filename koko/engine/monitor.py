"""Per-event latency tracing and periodic stage health reporting."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from koko.engine.config import Config
from koko.engine.events import Event

log = logging.getLogger("koko.monitor")


class Monitor:
    def __init__(self, cfg: Config, every_s: float = 10.0):
        self.cfg = cfg
        self.every_s = every_s
        self._hist: deque[float] = deque(maxlen=1000)

    def observe(self, ev: Event) -> None:
        self._hist.append(time.monotonic() - ev.ts)

    async def run_report_loop(self, stop=None) -> None:
        while True:
            try:
                await asyncio.sleep(self.every_s)
            except asyncio.CancelledError:
                return
            if stop is not None and stop.is_set():
                return
            if not self._hist:
                continue
            samples = sorted(self._hist)
            n = len(samples)
            p50 = samples[n // 2]
            p95 = samples[int(n * 0.95)]
            log.info("event age p50=%.3f p95=%.3f n=%d", p50, p95, n)
