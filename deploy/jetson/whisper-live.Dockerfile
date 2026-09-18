ARG BASE_IMAGE=docker.io/dustynv/whisperx:r35.3.1
FROM ${BASE_IMAGE}

WORKDIR /app

COPY deploy/jetson/requirements.whisper-live.txt /tmp/requirements.whisper-live.txt

# JetPack 5's R35 images expose Python 3.8, while WhisperLive 0.10 and its
# required faster-whisper 1.2.0 declare Python >=3.9. Their server code is
# usable here; bypass only the package metadata check. Keep --no-deps so pip
# does not replace Jetson's CTranslate2/CUDA stack with incompatible wheels.
# The referenced install guide's faster-whisper 1.2.1 cannot be used here:
# WhisperLive 0.10.0 pins faster-whisper==1.2.0.
RUN python3 -m pip install --no-cache-dir tomli \
    --ignore-requires-python --no-deps \
    "whisper-live==0.10.0" \
    "faster-whisper==1.2.0"

# WhisperLive's --no-deps install above intentionally skips its dependency
# resolver. Add only its HTTP/WebSocket runtime dependencies; the GPU/ML
# dependencies must continue to come from the Jetson base image.
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.whisper-live.txt

COPY engine/config.py ./engine/config.py
COPY koko/whisper_live_server.py ./koko/whisper_live_server.py

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
