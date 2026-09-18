ARG BASE_IMAGE=dustynv/whisperx:r35.3.1
FROM ${BASE_IMAGE}

WORKDIR /app

# whisperx already contains Jetson-built faster-whisper/CTranslate2. Installing
# WhisperLive without dependencies prevents pip from replacing it with x86/CUDA wheels.
RUN python3 -m pip install --no-cache-dir tomli "whisper-live==0.10.0" --no-deps

COPY engine/config.py ./engine/config.py
COPY koko/whisper_live_server.py ./koko/whisper_live_server.py

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
