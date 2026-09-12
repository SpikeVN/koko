# VieNeu TTS on Xavier: get RTF below 1.0

## Fresh measurements (this device, 60-frame synth, cProfile)

RTF 2.31 → 92 ms/frame (25 fps). Breakdown per generated frame:

| component | ms/frame | share |
|---|---|---|
| 16× `sess_ac.run` (1-layer acoustic, per RVQ codebook) | ~56 | 61% |
| numpy sampling (`samp` head matmul + `_sample` + `np.random.choice`) | ~28 | 30% |
| `sess_dec.run` (12-layer backbone decode_step) | ~3.3 | 4% |
| prefill | 1 call/utterance | — |

Critical finding: **everything runs on CPUExecutionProvider**. The PyPI aarch64
wheel (onnxruntime 1.19.2) has no CUDA EP — the Volta GPU is idle. Meanwhile
CUDA 11.4 toolkit, cuDNN 8.6, and TensorRT **8.5.2 runtime libs are installed**
(`libnvinfer.so.8`, builder, plugins — dpkg-verified).

**Why the TRT blocker dissolves:** the missing `libcudla.so.1` only affects
`python3-libnvinfer` (TensorRT's *Python API*). ORT's TRT execution provider
links `libnvinfer.so.8` directly. We never needed the TRT python bindings —
we need an ORT build with CUDA/TRT EPs for aarch64 py3.8, which NVIDIA ships
as Jetson wheels.

## Phase 1 — install NVIDIA's ORT GPU wheel (JetPack 5.1.2 / py3.8 / CUDA 11.4)

1. Harvest candidate wheel URLs (channels reachable from this device):
   - NVIDIA forum threads via curl + Discourse JSON API (`/t/<id>.json`),
     e.g. thread 283946 "CudaExecutionProvider doesn't appear on onnxruntime
     1.15.1" and adjacent JP5 ORT threads → `nvidia.box.com/shared/static/*.whl`
     links. Test-download each (they do rot — one already 404'd).
   - elinux Jetson Zoo `#ONNX_Runtime` section (Anubis-blocked for us; if all
     forum harvests fail, ask you to open it in a browser and paste links).
   - **Fallback:** build ORT 1.17.1 from source on-device (cmake + CUDA EP,
     ~1–2 h on the 6 Carmel cores; no cargo involved — that constraint was
     about a Rust dep, not ORT).
2. Install over the CPU wheel via uv; keep `onnxruntime==1.19.2` documented
   as the CPU fallback (env-switchable, see Phase 2).
3. Verify `get_available_providers()` → `CUDAExecutionProvider` (and
   `TensorRTExecutionProvider` if the wheel bundles it).

## Phase 2 — wire CUDA EP into `engine/tts_vieneu.py`

- New env knob `KOKO_TTS_EP=auto|cuda|cpu` (default auto = CUDA if available,
  else CPU) + provider options (device_id, arena) in a small `_providers()`
  helper; all four sessions go through it.
- **IOBinding for `sess_dec` + prefill**: the 12-layer KV cache is both input
  and output, full-length each frame. On GPU that must stay resident as device
  `OrtValue`s (bind inputs/outputs, swap references) — without this the
  host↔device round-trip per frame makes GPU net-negative.
- `sess_ac`: tiny graph, 17 runs/frame — measure plain `run()` vs IOBinding on
  GPU; keep whichever is faster. If GPU launch overhead dominates, leaving
  acoustic on CPU is acceptable.
- Correctness gate: argmax run (temperature 0) on CPU vs GPU must produce
  identical frames; stochastic mode spot-checked by ear + waveform sanity.

## Phase 3 — numpy sampling micro-opts (do regardless of GPU)

- Replace `np.random.choice(p=...)` with cumsum + `searchsorted` + a single
  uniform draw (0.23 s per 60 frames today).
- Kill redundant `astype`/`copy` in `_sample`/`samp`; preallocate buffers.
- Expected: 28 → ~10 ms/frame. Pure `engine/tts_vieneu.py` changes, no deps.

## Phase 4 (stretch) — TRT EP for prefill + decode_step only

If the wheel bundles the TRT EP: try fp16 engines for the two backbone graphs
(engine cache dir on disk), acoustic stays CUDA/CPU. Keep only if it beats
CUDA EP in the benchmark. (No `python3-libnvinfer`, no libcudla involved.)

## Success criterion

RTF < 1.0 on the 60-frame benchmark (< 40 ms/frame), unchanged audio
(argmax-identical). Projected: acoustic 56→8-15 ms, sampling ~10 ms,
decode ~1-2 ms ⇒ ~20-27 ms/frame ≈ RTF 0.5-0.7.

## Out of scope

- `onnx_update/` graphs (pending, unexplained 4× larger backbone data — separate work)
- codec decoder sessions (fast already; revisit only if the new profile says so)
- voice cloning / speaker encoder, ASR/LLM stages
