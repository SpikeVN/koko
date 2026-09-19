"""Simple in-process event router.

Design choice: one bus, many queues out. Each stage subscribes to the
kinds it cares about and gets its own independent asyncio.Queue, so a
slow consumer never blocks a fast producer (backpressure is per-stage).

Stages running on non-loop threads publish via `publish_threadsafe`,
which is drained by the bus pump task.
"""

from __future__ import annotations

import asyncio
import queue
import logging
import threading

from koko.engine.events import Event, Kind

log = logging.getLogger("koko.bus")


class Bus:
    def __init__(self, maxsize: int = 512):
        self._maxsize = maxsize
        self._subs: dict[Kind, list[asyncio.Queue[Event]]] = {}
        self._lock = threading.Lock()
        # inbound bridge for producers on other threads
        self._inbound: queue.SimpleQueue[Event] = queue.SimpleQueue()

    def subscribe(self, *kinds: Kind) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._maxsize)
        with self._lock:
            for k in kinds:
                self._subs.setdefault(k, []).append(q)
        return q

    def put(self, ev: Event) -> None:
        try:
            self._inbound.put_nowait(ev)
        except queue.Full:
            log.warning("inbound overflow, dropping %s", ev.kind)

    async def publish(self, ev: Event) -> None:
        # direct in-loop publish (no bridge hop)
        with self._lock:
            queues = list(self._subs.get(ev.kind, ()))
        for q in queues:
            if q.full():
                log.warning("queue overflow dropping %s", ev.kind)
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(ev)

    async def pump(self, stop: "asyncio.Event") -> None:
        """Drain inbound thread-safe queue into regular publish loop."""
        while not stop.is_set():
            try:
                ev = await asyncio.to_thread(self._inbound.get, timeout=0.2)
            except queue.Empty:
                continue
            await self.publish(ev)


async def get_or_stop(q: "asyncio.Queue[Event]", stop: asyncio.Event) -> Event | None:
    """Queue.get() that also returns when `stop` is set (returns None)."""
    get_task = asyncio.create_task(q.get())
    stop_task = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({get_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    if stop.is_set():
        get_task.cancel()
        return None
    return get_task.result()
