"""Component-timing benchmark for the VieNeu TTS frame loop.

Usage: .venv/bin/python tools/bench_tts.py [max_frames] [runs]
Prints ms/frame for: prefill, sess_dec (12-layer backbone), sess_ac+numpy
heads (acoustic frame loop), rest (embedding/sampling glue) and end-to-end RTF.
"""
import logging
import sys
import time

import numpy as np

# Vieneu frame rate: one autoregressive frame = 1/25 s of audio
FRAME_HZ = 25.0

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from engine.tts_vieneu import VieneuLite  # noqa: E402

PH = ("xin chào, đây là một đoạn văn dài hơn dùng để đo tốc độ tổng hợp "
      "giọng nói tiếng việt, hy vọng nó đủ nhanh cho realtime")


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    t0 = time.perf_counter()
    eng = VieneuLite("tts_model")
    print("load: %.1fs" % (time.perf_counter() - t0))

    # warm up sessions fully
    list(eng.stream_frames("xin chào", max_new_frames=8))

    stats = {"ac": 0.0, "dec": 0.0, "samp": 0.0}
    dec_run = eng.sess_dec.run
    pre_run = eng.sess_pre.run

    # instrument the backbone decode step
    def dec_timed(names, feed, ro=None):
        t = time.perf_counter()
        out = dec_run(names, feed, ro)
        stats["dec"] += time.perf_counter() - t
        return out
    eng.sess_dec.run = dec_timed

    blob = [0.0]  # non-ORT glue time per frame
    orig_ac = VieneuLite._acoustic_frame

    def ac_timed(self, h, *a, **k):
        glue0 = time.perf_counter()
        r = orig_ac(self, h, *a, **k)
        dt = time.perf_counter() - glue0
        stats["ac"] += dt
        return r
    VieneuLite._acoustic_frame = ac_timed

    for run in range(n_runs):
        for k in stats:
            stats[k] = 0.0
        t0 = time.perf_counter()
        frames = list(eng.stream_frames(PH, max_new_frames=n_frames))
        total = time.perf_counter() - t0
        nf = len(frames)
        audio_s = nf / FRAME_HZ
        dec_ms = stats["dec"] / nf * 1000
        ac_ms = stats["ac"] / nf * 1000
        pre_ms = (total - stats["ac"] - stats["dec"]) / nf * 1000
        print("run%d: %d frames (%.1fs audio), %.1f ms/frame, RTF %.2f | "
              "backbone %.1f | acoustic+numpy %.1f | glue %.1f ms/frame"
              % (run, nf, audio_s, total / nf * 1000, total / audio_s,
                 dec_ms, ac_ms, pre_ms))
        # sanity: components should sum to the printed ms/frame
        print("      components (backbone+acoustic+glue): %.1f ms -> RTF %.2f"
              % (dec_ms + ac_ms + pre_ms, (dec_ms + ac_ms + pre_ms) / 1000 * FRAME_HZ))


if __name__ == "__main__":
    main()
