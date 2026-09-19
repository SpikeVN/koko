ARG BASE_IMAGE=ghcr.io/spikevn/faster-whisper:jp5
FROM ${BASE_IMAGE}

WORKDIR /app

COPY deploy/jetson/requirements.whisper-live.txt /tmp/requirements.whisper-live.txt

# JetPack 5's R35 images expose Python 3.8, while WhisperLive 0.10 and its
# required faster-whisper version declare Python >=3.9. Their server code is
# usable here; bypass only the WhisperLive metadata check. Keep --no-deps so
# its installer uses the native Faster-Whisper/CTranslate2 CUDA stack from the
# JP5 base. Do not replace that stack with PyPI CUDA wheels.
RUN python3 -m pip install --no-cache-dir tomli \
    --ignore-requires-python --no-deps \
    "whisper-live==0.10.0"

# WhisperLive's --no-deps install above intentionally skips its dependency
# resolver. Add only its HTTP/WebSocket runtime dependencies; the GPU/ML
# dependencies must continue to come from the Jetson base image.
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.whisper-live.txt

COPY koko ./koko

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
