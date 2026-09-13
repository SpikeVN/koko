#!/usr/bin/env bash
# Provision all TTS model weights from Hugging Face (hf CLI).
#
# Sources of truth:
#   pnnbao-ump/VieNeu-TTS-v3-Turbo      -> tts_model/  (graph + heads + speaker/denoiser)
#     - onnx_update/ is the fp32 graph set; decode_step + acoustic + prefill
#     - The repo dir "onnx_int8/" holds the ORIGINAL MatMulInteger int8 graphs,
#     - which ORT's CUDA EP silently cannot run — do NOT use for the backbone.
#   OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX -> tts_model/codec/
#
# Not downloaded here (committed in git): voices_v3_turbo.json, config.json,
# tokenizer.json inside onnx_int8/, codec_browser_onnx_meta.json.
# Wheels (whls/) come from NVIDIA's GitHub releases, not HF — see MODELS.md.
#
# HF cache: on this box the root FS (`/`) is small, so the download cache goes
# to /mnt/sdcard/PhuongBase/downloads/hf_cache (override with HF_CACHE_DIR).
#
# Usage: tools/fetch_models.sh [--force]   (default: skips existing files)
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE="${HF_CACHE_DIR:-${KOKO_HF_CACHE:-$( [ -d /mnt/sdcard/PhuongBase/downloads ] && echo /mnt/sdcard/PhuongBase/downloads/hf_cache || echo "$PWD/hf_cache" )}}"
export HF_HOME="$CACHE" HF_HUB_CACHE="$CACHE"
mkdir -p "$CACHE"
VIE=$( [ "${1:-}" = "--force" ] && echo --force-download || echo )

echo "== Vieneu v3 Turbo: fp32 graphs (from onnx_update/) =="
tmp=$(mktemp -d)
hf download pnnbao-ump/VieNeu-TTS-v3-Turbo \
  --include "onnx_update/*" $VIE \
  --local-dir "$tmp" >/dev/null
mkdir -p tts_model/onnx_int8
mv -v "$tmp"/onnx_update/* tts_model/onnx_int8/
rm -rf "$tmp"

echo "== Vieneu: speaker encoder + denoiser (unwired, optional) =="
hf download pnnbao-ump/VieNeu-TTS-v3-Turbo \
  # note: each --include takes exactly one pattern; a second bare positional
  # would be parsed as a single-file FILENAME arg and silently skip the rest
  --include "speaker_encoder.onnx" --include "denoiser.onnx" \
  $VIE --local-dir tts_model >/dev/null

echo "== MOSS audio tokenizer (codec) =="
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX \
  --include "moss_audio_tokenizer_*" --include "codec_browser_onnx_meta.json" $VIE \
  --local-dir tts_model/codec >/dev/null

echo "done. sanity check:"
ls -la tts_model/ tts_model/onnx_int8/ tts_model/codec/ 2>/dev/null
