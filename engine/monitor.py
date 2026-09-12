"""Per-event latency tracing + periodic stage health report.

Every event carries a monotonic creation ts; stages stamp `marks` on it
on receipt/departure, and the monitor aggregates percentiles so "the bot
feels laggy" turns into numbers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from engine.config import Config
from engine.events import Event, Kind

log = logging.getLogger("koko.monitor")


class Monitor:
    def __init__(self, cfg: Config, every_s: float = 10.0):
        self.cfg = cfg
        self.every_s = every_s
        self._hist: deque[float] = deque(maxlen=1000)  # event age at first stage it is late at

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
