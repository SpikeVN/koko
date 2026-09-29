FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        portaudio19-dev \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# WhisperLive declares Python >=3.9, although its Faster-Whisper server works
# with Ubuntu 20.04's Python 3.8. CTranslate2 3.24.0 is the final CUDA 11/cuDNN
# 8 wheel, so replace WhisperLive's CUDA 12-oriented dependency after install.
RUN python3 -m pip install --ignore-requires-python whisper-live==0.10.0 \
    && python3 -m pip install --force-reinstall --no-deps ctranslate2==3.24.0

COPY koko ./koko

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
