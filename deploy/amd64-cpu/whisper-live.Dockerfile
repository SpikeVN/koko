FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential portaudio19-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install WhisperLive without its CUDA Torch dependencies, then install the
# CPU-only PyTorch and Faster-Whisper runtimes explicitly.
RUN python3 -m pip install --break-system-packages \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio \
    && python3 -m pip install --break-system-packages \
        --no-deps whisper-live==0.10.0 \
        faster-whisper==1.2.0 ctranslate2==4.6.0 \
        av tokenizers huggingface-hub onnxruntime \
        numpy websockets fastapi uvicorn python-multipart websocket-client \
        PyAudio tqdm requests

COPY koko ./koko

EXPOSE 9090
CMD ["python3", "-m", "koko.whisper_live_server", "--config", "/app/config.toml"]
