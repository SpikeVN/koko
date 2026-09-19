import asyncio
import unittest

from koko.engine.config import Config
from koko.engine.tts import VieneuTts
from koko.server import WsServer


class _WebSocket:
    remote_address = ("127.0.0.1", 0)


class _ConcurrentServer(WsServer):
    def __init__(self):
        super().__init__(Config())
        self.entered = 0
        self.both_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def _session(self, ws) -> None:
        self.entered += 1
        if self.entered == 2:
            self.both_entered.set()
        await self.release.wait()


class _TtsParent:
    sample_rate = 48_000

    def __init__(self):
        self.validated = []

    def _validate_voice(self, voice: str) -> None:
        self.validated.append(voice)


class MultiClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_connections_enter_sessions_concurrently(self):
        server = _ConcurrentServer()
        first = asyncio.create_task(server._wrapped(_WebSocket()))
        second = asyncio.create_task(server._wrapped(_WebSocket()))

        await asyncio.wait_for(server.both_entered.wait(), timeout=0.2)
        self.assertEqual(server.entered, 2)

        server.release.set()
        await asyncio.gather(first, second)

    def test_voice_selectors_are_connection_local(self):
        parent = _TtsParent()
        first = VieneuTts.Voice(parent, "Adam")
        second = VieneuTts.Voice(parent, "Bella")

        first.set_voice("Cora")

        self.assertEqual(first.voice, "Cora")
        self.assertEqual(second.voice, "Bella")
        self.assertEqual(parent.validated, ["Adam", "Bella", "Cora"])
