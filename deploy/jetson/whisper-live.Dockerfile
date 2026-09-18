ARG BASE_IMAGE=dustynv/whisperx:r35.3.1
FROM ${BASE_IMAGE}

WORKDIR /app

# JetPack 5's R35 images expose Python 3.8, while WhisperLive 0.10 declares
# Python >=3.9. Its server code is usable here; bypass only the package
# metadata check. Keep --no-deps so pip does not replace Jetson's
# faster-whisper/CTranslate2/CUDA stack with incompatible wheels.
RUN python3 -m pip install --no-cache-dir tomli \
    --ignore-requires-python "whisper-live==0.10.0" --no-deps

# WhisperLive's --no-deps install above intentionally skips its dependency
# resolver. Add only its HTTP/WebSocket runtime dependencies; the GPU/ML
# dependencies must continue to come from the Jetson base image.
RUN python3 -m pip install --no-cache-dir \
    "fastapi==0.103.2" \
    "uvicorn==0.23.2" \
    "python-multipart==0.0.6" \
    "websocket-client==1.6.4" \
    "websockets==11.0.3"

COPY engine/config.py ./engine/config.py
COPY koko/whisper_live_server.py ./koko/whisper_live_server.py

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
