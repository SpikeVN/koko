"""Minimal koko client: mic -> ws_server.py -> translated audio out.

  python ws_client.py [ws://host:6942] [--language en]

Sends raw mono PCM16 @ 16 kHz from the default mic; prints ASR status;
plays PCM16 48 kHz TTS chunks out the speakers. ctrl-c to quit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import websockets

IN_RATE = 16_000
OUT_RATE = 48_000
SLICE_MS = 20


def _capture(q: "queue.SimpleQueue[bytes]", stop_evt: threading.Event) -> None:
    """Thread: mic int16 capture into q."""

    def on_audio(indata, frames, time_info, status):
        q.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=IN_RATE, channels=1, dtype="int16",
        blocksize=int(IN_RATE * SLICE_MS / 1000), callback=on_audio)
    with stream:
        while not stop_evt.is_set():
            sd.sleep(50)


def _play(q: "queue.SimpleQueue[np.ndarray]", stop_evt: threading.Event) -> None:
    """Thread: drains decoded PCM float chunks out the speakers."""
    stream = None
    while not stop_evt.is_set():
        try:
            x = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if stream is None:
            stream = sd.OutputStream(samplerate=OUT_RATE, channels=1,
                                     dtype="float32")
            stream.start()
        stream.write(np.asarray(x, dtype=np.float32))


async def _amain(url: str, language: str) -> None:
    pcm_in: queue.SimpleQueue = queue.SimpleQueue()
    pcm_out: queue.SimpleQueue = queue.SimpleQueue()
    stop_evt = threading.Event()

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "language": language}).encode("utf-8"))
        capture_t = threading.Thread(target=_capture, args=(pcm_in, stop_evt), daemon=True)
        play_t = threading.Thread(target=_play, args=(pcm_out, stop_evt), daemon=True)
        capture_t.start()
        play_t.start()
        print("ready; speaking into mic... ctrl-c to stop")
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    x = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                    pcm_out.put(x)
                    continue
                ctl = json.loads(msg)
                t = ctl.get("type")
                if t in ("partial", "final"):
                    print("[%s] %s" % (t, ctl.get("text", "")))
                elif t == "error":
                    print("[server] %s" % ctl.get("detail"))
        finally:
            stop_evt.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="ws://127.0.0.1:6942")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()
    try:
        asyncio.run(_amain(args.url, args.language))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
