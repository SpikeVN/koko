"""Micro-bench: sess_ac thread configs + _sample internals."""
import logging
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
import onnxruntime as ort  # noqa: E402
from engine.tts_vieneu import VieneuLite  # noqa: E402

eng = VieneuLite("tts_model")

# --- sess_ac configs ------------------------------------------------------
PAST = 16
pk = [np.random.randn(1, 8, PAST, 96).astype(np.float32)] * eng.L_loc
pv = [np.random.randn(1, 8, PAST, 96).astype(np.float32)] * eng.L_loc
tok = np.random.randn(1, 1, eng.hidden).astype(np.float32)
feed = {"token_emb": tok, "position_ids": np.array([[9]], np.int64),
        **eng._past_feed(pk, pv)}

def bench_sess(threads, spin):
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", spin)
    s = ort.InferenceSession("tts_model/onnx_int8/vieneu_acoustic_cached.onnx",
                             so, providers=["CPUExecutionProvider"])
    for _ in range(50):
        s.run(None, feed)
    t0 = time.time()
    for _ in range(300):
        s.run(None, feed)
    return (time.time() - t0) / 300 * 1000

for th, spin in ((4, "0"), (4, "1"), (1, "0"), (1, "1"), (2, "1")):
    print("sess_ac intra=%d spin=%s : %.3f ms/run  (x16 = %.1f ms/frame)"
          % (th, spin, bench_sess(th, spin), bench_sess(th, spin) * 16))

# --- _sample internals ----------------------------------------------------
logits = np.random.randn(1024).astype(np.float32) * 5
prev_set = set(np.random.randint(0, 1024, 60).tolist())
hist = VieneuLite("tts_model")  # reuse engine's _sample

def bench(f, n=2000):
    f()
    t0 = time.time()
    for _ in range(n):
        f()
    return (time.time() - t0) / n * 1e6

# full sample call
print("_sample full   : %.0f us (x16 = %.1f ms/frame)"
      % (bench(lambda: eng._sample(logits.copy(), 0.8, 25, 0.95, 1.2, prev_set)),
         bench(lambda: eng._sample(logits.copy(), 0.8, 25, 0.95, 1.2, prev_set)) * 16 / 1000))

# pieces
c = eng.cfg
lg = logits / 0.8
piece_clone = lambda t, n=500: print("  " + t)
print("  fromiter      : %.0f us" % bench(lambda: np.fromiter(prev_set, np.int64, count=60)))
idx = np.fromiter(prev_set, dtype=np.int64, count=60)
sel = logits[idx]
print("  penalty branch: %.0f us" % bench(lambda: np.where(sel < 0, sel * 1.2, sel / 1.2)))
print("  argpartition  : %.0f us" % bench(lambda: np.argpartition(lg, -25)[-25:]))
cand = np.argpartition(lg, -25)[-25:]
cs = lg[cand]
print("  argsort-rev   : %.0f us" % bench(lambda: np.argsort(cs)[::-1]))
o = np.argsort(cs)[::-1]
cand2 = cand[o]
p = (cs[o] - cs[o].max())
e = np.exp(p)
p = e / e.sum()
cum = p.cumsum()
r = 0.5
print("  searchsorted  : %.0f us" % bench(lambda: cum.searchsorted(r, side="right")))
print("  np.where pass : %.0f us" % bench(lambda: cum.searchsorted(np.random.random_sample())))
