FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

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

# CUDA 12 and cuDNN 9 are supplied by the NVIDIA base image. WhisperLive pins
# faster-whisper and lets it install a current CTranslate2 build for that ABI.
# PyAudio is not used by this server, but WhisperLive declares it as a runtime
# dependency and needs PortAudio headers when no wheel is available.
RUN python3 -m pip install --break-system-packages whisper-live==0.10.0

COPY koko ./koko

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
