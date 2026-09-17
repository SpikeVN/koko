"""API-only asynchronous client for the koko websocket protocol.

This module contains the reusable protocol layer.  It deliberately does not
open a microphone, create a speaker stream, resample audio, or print UI
output.  Applications provide 16-bit microphone frames to :meth:`send_audio`
and decide how to consume the audio frames yielded by :meth:`messages`.

Protocol summary
----------------

Koko uses one websocket for both directions.  Client-to-server binary frames
are raw mono PCM16 little-endian audio at 16 kHz.  Client-to-server text
frames are JSON control objects.  Server-to-client text frames are JSON
events; server-to-client binary frames are raw mono PCM16 output audio.

An output audio frame is always preceded by an ``{"type": "audio",
"rate": N}`` event.  The rate belongs to the next binary frame, not to the
whole connection.  :class:`KokoClient` remembers that announcement and
returns it as :attr:`KokoAudio.rate` on the following binary message.

Minimal usage::

    async def run() -> None:
        client = KokoClient("ws://127.0.0.1:6942")
        await client.connect(language="en")
        try:
            await client.send_audio(pcm16_frame)
            async for message in client.messages():
                if isinstance(message, KokoAudio):
                    play_or_queue(message.pcm16, message.rate)
                elif message.type == "translation":
                    print(message.data["text"])
        finally:
            await client.close()

The receive loop should normally run for the entire session.  Audio and text
events are interleaved, and failing to consume the websocket can prevent
audio and control messages from being processed.  The server allows only one
live session, so callers should reuse one connected instance rather than
opening parallel connections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np
import websockets


@dataclass(frozen=True)
class KokoControl:
    """A JSON control message received from koko.

    Attributes:
        type: The server event name, such as ``ready``, ``partial``,
            ``translation``, ``audio``, or ``error``.
        data: The complete decoded JSON object.  Keeping the full object makes
            the client forward-compatible with new server fields.

    Common fields include:

    * ``ready``: ``asr_language``, ``asr_auto_detect``, ``tts_voices``, and
      ``tts_voice``.
    * ``partial``, ``final``, ``speak``, ``translation``: ``text``.
    * ``error``: ``detail``.
    * ``audio``: ``rate`` for the binary frame that follows.
    """

    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class KokoAudio:
    """One server-to-client mono PCM16 little-endian audio frame.

    The bytes are not an encoded file and have no WAV header.  They are
    signed 16-bit little-endian samples at :attr:`rate`.  A frame can be
    written immediately to an output device configured for that rate.  The
    rate may change during one session, so an application should recreate or
    reconfigure its output stream when it changes.

    Attributes:
        pcm16: Raw PCM bytes.  The length should be even because every sample
            occupies two bytes.
        rate: Sample rate announced immediately before this frame.
    """

    pcm16: bytes
    rate: int

    def samples(self) -> np.ndarray:
        """Return normalized samples as a new ``float32`` NumPy array.

        Conversion divides signed PCM values by 32768, matching the common
        audio callback representation.  This method does not resample, copy
        channel data, or perform playback.  Use :attr:`pcm16` directly when a
        playback API accepts PCM16 bytes.
        """
        return np.frombuffer(self.pcm16, dtype=np.int16).astype(np.float32) / 32768.0


KokoMessage = KokoControl | KokoAudio


class KokoClient:
    """Small protocol client with no audio-device or UI dependencies.

    The object has a simple lifecycle: construct, :meth:`connect`, exchange
    audio and events, then :meth:`close`.  It is not a context manager because
    callers commonly need to coordinate device threads and websocket tasks;
    the explicit lifecycle makes that coordination visible.

    Only one consumer should iterate :meth:`messages` at a time.  Sending may
    happen from the same asyncio event loop while that iterator is running.
    The class does not impose a queue limit on received messages; applications
    should process or forward audio promptly rather than accumulating it.
    """

    def __init__(self, url: str = "ws://127.0.0.1:6942") -> None:
        """Create a disconnected client for ``url``.

        Args:
            url: Websocket endpoint for the koko server.  The default matches
                the local server command.
        """
        self.url = url
        self._ws: Any = None
        self._audio_rate = 48_000

    @property
    def connected(self) -> bool:
        """Whether a websocket has been established for this instance.

        This reports client state, not whether the remote peer is healthy at
        this exact instant.  A later send or receive can still raise after a
        network failure has occurred.
        """
        return self._ws is not None

    async def connect(
        self,
        language: str = "en",
        *,
        auto_detect: bool = False,
    ) -> None:
        """Connect and send the initial ``hello`` control frame.

        Args:
            language: Initial Whisper source-language code, for example
                ``"en"`` or ``"vi"``.
            auto_detect: Whether the server should enable Whisper language
                auto-detection for this session.

        Raises:
            RuntimeError: If this instance is already connected.
            OSError or a websocket exception: If the connection fails.

        The server's ``ready`` response is delivered later through
        :meth:`messages`; successful return from this method means the hello
        frame was sent, not that ``ready`` has already been received.
        """
        if self._ws is not None:
            raise RuntimeError("koko client is already connected")
        self._ws = await websockets.connect(self.url, max_size=None)
        try:
            await self._send_control({
                "type": "hello",
                "language": language,
                "asr_auto_detect": auto_detect,
            })
        except BaseException:
            await self.close()
            raise

    async def close(self, *, send_bye: bool = True) -> None:
        """Close the websocket and make the client reusable.

        Args:
            send_bye: Send the protocol ``bye`` control frame before closing.
                Set this to ``False`` during failure cleanup when the socket
                may already be unusable.

        Closing is idempotent.  It does not drain queued server audio; callers
        that need a graceful final playback should continue consuming
        :meth:`messages` before calling this method.
        """
        ws, self._ws = self._ws, None
        if ws is None:
            return
        if send_bye:
            try:
                await ws.send(json.dumps({"type": "bye"}))
            except websockets.ConnectionClosed:
                pass
        await ws.close()

    async def send_audio(self, pcm16: bytes | bytearray | memoryview) -> None:
        """Send one raw microphone frame to the server.

        Args:
            pcm16: Mono signed PCM16 little-endian bytes sampled at 16 kHz.
                Frames around 20 ms (640 bytes) are recommended.  The method
                accepts bytes-like objects and sends a detached ``bytes``
                object, so later mutation of a bytearray is safe.

        The payload must contain audio only.  Do not add a WAV header, JSON
        envelope, or sample-rate prefix.  The server may drop input under
        downstream pressure; callers should keep their capture queue shallow.
        """
        await self._require_ws().send(bytes(pcm16))

    async def set_language(self, language: str) -> None:
        """Request a new Whisper source language for the active session."""
        await self._send_control({"type": "asr_language", "language": language})

    async def set_auto_detect(self, enabled: bool) -> None:
        """Enable or disable server-side automatic source-language detection."""
        await self._send_control({"type": "asr_auto_detect", "enabled": enabled})

    async def set_voice(self, voice: str) -> None:
        """Request a TTS voice preset by its exact server-provided name."""
        await self._send_control({"type": "tts_voice", "voice": voice})

    async def clear_context(self) -> None:
        """Discard buffered source text and the server's LLM conversation history."""
        await self._send_control({"type": "clear_context"})

    async def messages(self) -> AsyncIterator[KokoMessage]:
        """Yield server messages in wire order until the socket closes.

        JSON frames become :class:`KokoControl`.  A binary frame becomes
        :class:`KokoAudio` using the sample rate from the most recent
        ``audio`` control event.  The ``audio`` announcement itself is also
        yielded as a :class:`KokoControl`, so consumers that need every server
        event can inspect it; consumers interested only in playable data can
        ignore controls whose ``type`` is ``"audio"``.

        Raises:
            RuntimeError: If called before :meth:`connect`.
            ValueError: If a text frame is not a JSON object with a string
                ``type`` field.
            websockets.exceptions.ConnectionClosed: When the peer closes
                unexpectedly, depending on the websocket library version.

        The iterator is single-consumer.  Do not run two independent
        ``async for`` loops over one client.
        """
        ws = self._require_ws()
        async for raw in ws:
            if isinstance(raw, bytes):
                yield KokoAudio(raw, self._audio_rate)
                continue
            data = json.loads(raw)
            if not isinstance(data, dict) or not isinstance(data.get("type"), str):
                raise ValueError("invalid koko control message")
            if data["type"] == "audio":
                self._audio_rate = int(data.get("rate", self._audio_rate))
            yield KokoControl(data["type"], data)

    async def _send_control(self, data: dict[str, Any]) -> None:
        await self._require_ws().send(json.dumps(data))

    def _require_ws(self) -> Any:
        if self._ws is None:
            raise RuntimeError("koko client is not connected")
        return self._ws
