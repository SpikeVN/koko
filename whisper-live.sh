#!/bin/bash
# Run the WhisperLive ASR server.
#
# whisper-live / faster-whisper pull in the plain `onnxruntime` distribution,
# which clashes with the GPU `onnxruntime-gpu` our TTS needs — the two can't
# share one venv. So whisper-live runs in its own dedicated venv
# (.venv-whisperlive), separate from the koko pipeline. This script creates
# and provisions that venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-whisperlive
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "== creating $VENV and installing whisper-live =="
    uv venv "$VENV"
    # openai-whisper (a whisper-live dep) builds against pkg_resources, which
    # setuptools >=81 dropped; install setuptools<81 and build without
    # isolation so the wheel builds.
    uv pip install --python "$PY" --no-build-isolation \
        "setuptools<81" "whisper-live"
fi

# Switchless: every ASR server setting (host, port, model, backend, max
# clients, connection cap) lives in config.toml [asr].
"$PY" -m koko.whisper_live_server
