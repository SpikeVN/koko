FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/amd64/requirements.websocket.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt

COPY koko ./koko
COPY tts_model/voices_v3_turbo.json ./tts_model/voices_v3_turbo.json
COPY tts_model/more_voices.json ./tts_model/more_voices.json

EXPOSE 6942
CMD ["python3", "-m", "koko.server", "--config", "/app/config.toml"]
