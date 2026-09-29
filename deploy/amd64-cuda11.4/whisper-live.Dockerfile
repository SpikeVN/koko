FROM docker.io/nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

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

# Like the Jetson image, install the package without its CUDA 12/Python 3.9+
# resolver constraints, then provide an explicit CUDA 11-compatible runtime.
COPY deploy/amd64-cuda11.4/requirements.whisper-live.txt /tmp/requirements.whisper-live.txt
RUN python3 -m pip install --ignore-requires-python --no-deps whisper-live==0.10.0 \
    && python3 -m pip install --no-deps ctranslate2==3.24.0 faster-whisper==0.10.1 \
    && python3 -m pip install -r /tmp/requirements.whisper-live.txt

COPY koko ./koko

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
