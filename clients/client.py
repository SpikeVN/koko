"""Minimal koko client: microphone -> websocket server -> translated audio.

  uv run koko-client [ws://host:6942] [--language en]

Sends raw mono PCM16 @ 16 kHz from the default mic; prints ASR status;
plays PCM16 48 kHz TTS chunks out the speakers. ctrl-c to quit.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading

import numpy as np
import websockets
import sounddevice as sd
from . import client_devices as cdev
from .koko_client import KokoAudio, KokoClient, KokoControl

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
          rate: list[int], device) -> None:
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
                                     dtype="float32", device=device)
            stream.start()
        stream.write(np.asarray(x, dtype=np.float32))
    if stream is not None:
        stream.stop()
        stream.close()


async def _amain(url: str, language: str, device, out_device) -> None:
    pcm_in: queue.SimpleQueue = queue.SimpleQueue()
    pcm_out: queue.SimpleQueue = queue.SimpleQueue()
    stop_evt = threading.Event()

    client = KokoClient(url)
    await client.connect(language)
    try:
        capture_t = threading.Thread(target=_capture, args=(pcm_in, stop_evt, device), daemon=True)
        out_rate = [OUT_RATE]
        play_t = threading.Thread(target=_play, args=(pcm_out, stop_evt, out_rate, out_device),
                                  daemon=True)
        capture_t.start()
        play_t.start()
        dev = sd.query_devices(device or None, kind="input")
        if device is None and cdev.pipewire_available():
            default = cdev.default_source()
            mic_name = default.name if default else dev["name"]
        else:
            mic_name = dev["name"]
        print("mic: %s" % mic_name)
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
                    await client.send_audio(raw)

        send_t = asyncio.ensure_future(_sender())
        try:
            async for msg in client.messages():
                if isinstance(msg, KokoAudio):
                    out_rate[0] = msg.rate
                    pcm_out.put(msg.samples())
                    continue
                assert isinstance(msg, KokoControl)
                if msg.type == "translation":
                    print("[translation] %s" % msg.data.get("text", ""), flush=True)
                elif msg.type == "speak":
                    print("[speak] %s" % msg.data.get("text", ""), flush=True)
                elif msg.type == "error":
                    print("[server] %s" % msg.data.get("detail"), flush=True)
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
    finally:
        await client.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="ws://127.0.0.1:6942")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default=None,
                    help="input device: index or substring of the name "
                         "(default: system default)")
    ap.add_argument("--out-device", default=None,
                    help="output device (headphones): index or substring "
                         "of the name (default: system default)")
    ap.add_argument("--list-devices", action="store_true",
                    help="list input devices and exit")
    ui = ap.add_mutually_exclusive_group()
    ui.add_argument("--ui", action="store_true",
                    help="run the Tkinter desktop client (default)")
    ui.add_argument("--no-ui", action="store_true",
                    help="headless client (no desktop window)")
    args = ap.parse_args()

    if args.list_devices:
        if cdev.pipewire_available():
            devs = sd.query_devices()
            print("Audio sources (mics)  [* = default]  [dev: N = PortAudio index]:")
            for d in cdev.list_sources(devs):
                idx = d.dev_index if d.dev_index is not None else "?"
                print("  %s %-4d dev:%s %s"
                      % ("*" if d.is_default else " ", d.id, idx, d.name))
            print("Audio outputs (speakers)  [* = default]  [dev: N = PortAudio index]:")
            for d in cdev.list_sinks(devs):
                idx = d.dev_index if d.dev_index is not None else "?"
                print("  %s %-4d dev:%s %s"
                      % ("*" if d.is_default else " ", d.id, idx, d.name))
        else:
            print(sd.query_devices())
        return

    # ---- device resolution: friendly name -> PortAudio device index
    device = out_device = None
    if cdev.pipewire_available():
        devs = sd.query_devices()
        for arg, kind, name in ((args.device, "source", "input"),
                                (args.out_device, "sink", "output")):
            if arg is None:
                continue
            if arg.isdigit():                     # explicit PortAudio index
                if kind == "source":
                    device = int(arg)
                else:
                    out_device = int(arg)
                continue
            pool = (cdev.list_sources(devs) if kind == "source"
                    else cdev.list_sinks(devs))
            hit = next((d for d in pool if arg.lower() in d.name.lower()), None)
            if hit is None:
                ap.error("no %s device matching %r; run with --list-devices"
                         % (name, arg))
            if hit.dev_index is None:
                ap.error("couldn't map %s device %r to a sound device"
                         % (name, hit.name))
            if kind == "source":
                device = hit.dev_index
            else:
                out_device = hit.dev_index
            print("%s: %s" % (name, hit.name))
    else:
        if args.device is not None and not args.device.isdigit():
            devs = sd.query_devices()
            match = [i for i, d in enumerate(devs) if d["name"]
                     and args.device.lower() in d["name"].lower()]
            if not match:
                ap.error("no input device matching %r; list with `python3 "
                          "koko-client --list-devices`" % args.device)
            device = match[0]
        elif args.device is not None:
            device = int(args.device)
        if args.out_device is not None and not args.out_device.isdigit():
            devs = sd.query_devices()
            match = [i for i, d in enumerate(devs) if d["name"]
                     and args.out_device.lower() in d["name"].lower()
                     and d["max_output_channels"] > 0]
            if not match:
                ap.error("no output device matching %r; list with `python3 "
                          "koko-client --list-devices`" % args.out_device)
            out_device = match[0]
        elif args.out_device is not None:
            out_device = int(args.out_device)

    use_ui = not args.no_ui
    if use_ui:
        from .client_ui import run_client_ui
        run_client_ui(args.url, args.language, device, out_device)
        return

    try:
        asyncio.run(_amain(args.url, args.language, device, out_device))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
