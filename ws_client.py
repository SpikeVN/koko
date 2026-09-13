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


def _capture(q: "queue.SimpleQueue[bytes]", stop_evt: threading.Event,
             device) -> None:
    """Thread: mic int16 capture into q."""

    def on_audio(indata, frames, time_info, status):
        raw = bytes(indata)               # cffi buffer -> bytes
        pct = int(np.max(np.abs(np.frombuffer(raw, dtype=np.int16))) / 327.67)
        print(f"\rmic level {pct:3d}%   ", end="", flush=True)
        q.put(raw)

    stream = sd.RawInputStream(
        samplerate=IN_RATE, channels=1, dtype="int16",
        blocksize=int(IN_RATE * SLICE_MS / 1000), callback=on_audio,
        device=device)
    with stream:
        while not stop_evt.is_set():
            sd.sleep(50)


def _play(q: "queue.SimpleQueue[np.ndarray]", stop_evt: threading.Event,
          rate: list[int]) -> None:
    """Thread: drains decoded PCM float chunks out the speakers.

    `rate` is a one-item list holding the server's current sample rate;
    the stream is (re)opened whenever it changes."""
    stream = None
    while not stop_evt.is_set():
        try:
            x = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if stream is None or stream.samplerate != rate[0]:
            if stream is not None:
                stream.stop()
                stream.close()
            stream = sd.OutputStream(samplerate=rate[0], channels=1,
                                     dtype="float32")
            stream.start()
        stream.write(np.asarray(x, dtype=np.float32))
    if stream is not None:
        stream.stop()
        stream.close()


async def _amain(url: str, language: str, device) -> None:
    pcm_in: queue.SimpleQueue = queue.SimpleQueue()
    pcm_out: queue.SimpleQueue = queue.SimpleQueue()
    stop_evt = threading.Event()

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "language": language}))
        capture_t = threading.Thread(target=_capture, args=(pcm_in, stop_evt, device), daemon=True)
        out_rate = [OUT_RATE]
        play_t = threading.Thread(target=_play, args=(pcm_out, stop_evt, out_rate), daemon=True)
        capture_t.start()
        play_t.start()
        dev = sd.query_devices(device or None, kind="input")
        print("mic: %s" % dev["name"])
        print("ready; speaking into mic... ctrl-c to stop")
        loop = asyncio.get_running_loop()

        async def _sender() -> None:   # drain captured PCM -> websocket
            def get_block() -> bytes:
                return pcm_in.get(timeout=0.2)
            while not stop_evt.is_set():
                try:
                    raw = await loop.run_in_executor(None, get_block)
                except queue.Empty:
                    continue
                if raw:
                    await ws.send(raw)

        send_t = asyncio.ensure_future(_sender())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    x = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                    pcm_out.put(x)
                    continue
                ctl = json.loads(msg)
                t = ctl.get("type")
                if t == "audio":                # sample-rate header before pcm
                    out_rate[0] = int(ctl.get("rate", OUT_RATE))
                    continue
                if t == "translation":
                    print("[translation] %s" % ctl.get("text", ""), flush=True)
                elif t == "speak":
                    print("[speak] %s" % ctl.get("text", ""), flush=True)
                elif t == "error":
                    print("[server] %s" % ctl.get("detail"), flush=True)
        finally:
            stop_evt.set()
            try:
                await send_t
            except (websockets.ConnectionClosed, asyncio.CancelledError):
                pass
            while not pcm_out.empty():        # unblock _play from stream.write
                try:
                    pcm_out.get_nowait()
                except queue.Empty:
                    break
            capture_t.join(timeout=1.0)
            play_t.join(timeout=1.0)   # portaudio segfaults if killed mid-write


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="ws://127.0.0.1:6942")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default=None,
                    help="input device: index or substring of the name "
                         "(default: system default)")
    ap.add_argument("--list-devices", action="store_true",
                    help="list input devices and exit")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    device = args.device
    if device is not None and not device.isdigit():
        devs = sd.query_devices()
        match = [i for i, d in enumerate(devs) if d["name"]
                 and device.lower() in d["name"].lower()]
        if not match:
            ap.error("no input device matching %r; list with `python3 "
                     "ws_client.py --list-devices`" % device)
        device = match[0]

    try:
        asyncio.run(_amain(args.url, args.language, device))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
