"""Instrument the real _acoustic_frame loop run-by-run.

Monkeypatches sess_ac.run_with_iobinding, the _ac_bind_* helpers and _sample
to attribute per-step cost. All per-step numbers are averages over frames.
"""
import logging
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from engine.tts_vieneu import VieneuLite  # noqa: E402

eng = VieneuLite("tts_model")
sess = eng.sess_ac
NFRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 50

MAXSTEPS = eng.n_vq
t_run = [0.0] * MAXSTEPS
t_bind = [0.0] * MAXSTEPS
t_samp = [0.0] * MAXSTEPS

orig_run = sess.run_with_iobinding
hist_run = []


def timed_run(binding, run_options=None):
    t0 = time.perf_counter()
    orig_run(binding, run_options)
    hist_run.append(time.perf_counter() - t0)


sess.run_with_iobinding = timed_run

orig_bind_in = VieneuLite._ac_bind_in
orig_bind_out = VieneuLite._ac_bind_out
hist_bind = []


def bi(self, *a, **k):
    t0 = time.perf_counter()
    r = orig_bind_in(self, *a, **k)
    hist_bind.append(time.perf_counter() - t0)
    return r


def bo(self, *a, **k):
    t0 = time.perf_counter()
    r = orig_bind_out(self, *a, **k)
    hist_bind.append(time.perf_counter() - t0)
    return r


orig_sample = VieneuLite._sample
hist_samp = []


def ts(self, *a, **k):
    t0 = time.perf_counter()
    r = orig_sample(self, *a, **k)
    hist_samp.append(time.perf_counter() - t0)
    return r


VieneuLite._ac_bind_in = bi
VieneuLite._ac_bind_out = bo
VieneuLite._sample = ts

arr = np.random.randn(1, eng.hidden).astype(np.float32)
for _ in range(6):
    eng._acoustic_frame(arr, 0.8, 25, 0.95, 1.2, None)

acc_run = [0.0] * eng.n_vq
acc_bind = [0.0] * eng.n_vq
acc_samp = [0.0] * eng.n_vq
cnt = NFRAMES
for _ in range(NFRAMES):
    hist_run.clear(); hist_bind.clear(); hist_samp.clear()
    eng._acoustic_frame(arr, 0.8, 25, 0.95, 1.2, None)
    for ch in range(eng.n_vq):
        acc_run[ch] += hist_run[ch] * 1000
        acc_bind[ch] += hist_bind[ch] * 1000
        acc_samp[ch] += hist_samp[ch] * 1000

print("per-step averages over %d frames:" % cnt)
run_tot = bind_tot = samp_tot = 0.0
for ch in range(eng.n_vq):
    r, b, s = acc_run[ch] / cnt, acc_bind[ch] / cnt, acc_samp[ch] / cnt
    run_tot += b + r + s
    print("  step %2d: run %.2f  bind %.2f  samp %.2f  (sum %.2f)"
          % (ch, r, b, s, r + b + s))
print("  frame total: %.1f ms" % run_tot)
