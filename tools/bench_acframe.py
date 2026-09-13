"""Time every phase of one _acoustic_frame call (16 codebook steps)."""
import logging
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from engine.tts_vieneu import VieneuLite  # noqa: E402

eng = VieneuLite("tts_model")
# spin=1 for acoustic like planned config
so = eng.sess_ac.get_session_options()

N = 100
hist = None
phases = {"pre": 0.0, "step0": 0.0, "steps": 0.0, "samp": 0.0, "eos": 0.0,
          "np": 0.0}

import engine.tts_vieneu as tv

# wrapper around _sample to time it
orig_sample = VieneuLite._sample
samp_time = [0.0]

def timed_sample(self, *a, **k):
    t0 = time.perf_counter()
    r = orig_sample(self, *a, **k)
    samp_time[0] += time.perf_counter() - t0
    return r
VieneuLite._sample = timed_sample

h = np.random.randn(1, eng.hidden).astype(np.float32)
for it in range(N):
    samp_time[0] = 0.0
    t0 = time.perf_counter()
    codes, eos = eng._acoustic_frame(h, 0.8, 25, 0.95, 1.2, None)
    total = time.perf_counter() - t0
    phases["samp"] += samp_time[0]
    phases["np"] += total - samp_time[0]
    if it == 0:
        print("frame total: %.1f ms" % (total * 1000))

print("over %d frames:" % N)
print("  ORT runs+glue in _acoustic_frame: %.1f ms/frame"
      % (phases["np"] / N * 1000))
print("  numpy _sample calls            : %.1f ms/frame"
      % (phases["samp"] / N * 1000))
