# Models & binaries to provision after clone/rsync

`.gitignore` / `sync` exclude the weights — git and rsync will not bring
these over. **Run `uv run tools/fetch_models.py`** to fetch everything from
Hugging Face; it caches downloads under
`/mnt/sdcard/PhuongBase/downloads/hf_cache` (root FS `/` lacks space).

## 1. One command

```bash
uv run tools/fetch_models.py            # skip files already present
uv run tools/fetch_models.py --force    # re-download everything
```

## 2. What it fetches

| HF repo | subpath/file | into |
|---|---|---|
| `pnnbao-ump/VieNeu-TTS-v3-Turbo` | `onnx_update/vieneu_*` (fp32 graphs+weights+heads) | `tts_model/onnx_int8/` |
| same | `speaker_encoder.onnx`, `denoiser.onnx` | `tts_model/` |
| `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX` | `moss_audio_tokenizer_*` | `tts_model/codec/` |

- `onnx_update/` is the **fp32** backbone set — the one in use since the
  MatMulInteger int8 graphs (`onnx_int8/` in that repo) were found to make
  ORT's CUDA EP silently fall back to CPU. The old int8 set is re-downloadable
  from the repo's `onnx_int8/` subfolder if an A/B comparison is ever needed.
- `tts_model/int8-era heads npz` note: before 2026-09-13 the
  `vieneu_v3_heads.npz` in local use was the one from the repo's `onnx_int8/`
  subfolder (int8-era, byte-identical to what we ran all along). The backup is
  at `/mnt/sdcard/PhuongBase/downloads/heads_local_backup.npz`; the script now
  provisions the `onnx_update` npz, which is the SDK-canonical pairing with
  the fp32 graphs (A/B tested 2026-09-13: both produce sane audio, e.g. peak
  0.83-0.88 / rms ~0.11 on a fixed test sentence).
- File sizes not touched by the script (committed in git):
  `voices_v3_turbo.json`, `onnx_int8/config.json`, `onnx_int8/tokenizer.json`,
  `codec_browser_onnx_meta.json`.

## 3. Wheels (`whls/`) — Jetson only, not on HF

| File | Size | Where |
|---|---|---|
| `onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl` | ~40 MB | `whls/` |
| `torch-1.14.0a0+44dac51c.nv23.01-cp38-cp38-linux_aarch64.whl` | ~573 MB | `whls/` |

- The ORT wheel is the **JetPack 5.1.1 NVIDIA build** (1.16.0) — PyPI and the
  1.18.0 builds don't work here. Download from NVIDIA's onnxruntime GitHub
  releases (`onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl`, jetpack
  5.1.1 variant).
- Install: `uv pip install --python .venv/bin/python whls/onnxruntime_gpu-1.16.0-*.whl`
  (uninstall any existing onnxruntime first).
- CUDA EP additionally needs `libcufft-11-4` via apt
  (`sudo apt install libcufft-11-4`); cufft/cublas/cudart are preloaded from
  system paths by `koko/engine/tts_vieneu.py`.
- Torch wheel only needed for re-exporting graphs; not required to run TTS.

## 4. Voices & remote services

- `tts_model/voices_v3_turbo.json` (~160 KB) is **committed** — no download.
- Phonemization is remote (sea-g2p has no cp38 wheel — the py310 aarch64 wheel
  is archived at `/mnt/sdcard/PhuongBase/downloads/seag2p/`): the server URL
  is configured via the `[phonemize]` section of `config.toml`. Ensure the
  LLM host exposes the phonemize endpoint before first synth.

## Quick sanity after provisioning

```bash
uv run tools/bench_tts.py
# expect (isolation, MAXN): backbone ~17 ms/frame, acoustic+numpy ~33,
# loop RTF ~1.2-1.3; with `tts.greedy = true` in config.toml a bit lower.
# Full pipeline RTF (incl. codec decode via thread) ~0.65-0.83 depending on
# box load.
```
