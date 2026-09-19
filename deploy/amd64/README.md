# AMD64 NVIDIA GPU deployment

This deployment targets Linux AMD64 hosts with an NVIDIA GPU, a driver that
supports CUDA 12.6, Docker, and NVIDIA Container Toolkit. It publishes the
same three-service stack as the Jetson deployment:

- `koko-server`: public websocket and Vieneu TTS service on port 6942.
- `koko-whisper`: CUDA Faster-Whisper ASR service on the private Compose network.
- `koko-phonemize`: CPU-only sea-g2p service on the private Compose network.

The GPU images use CUDA 12.6 with cuDNN 9. The host driver supplies `libcuda`;
do not install a CUDA toolkit in the containers or mount host CUDA libraries.

## Start

1. Install Docker and NVIDIA Container Toolkit, then verify GPU access:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
   ```

2. Fetch the required TTS files into `tts_model/` with `tools/fetch_models.py`.
3. Edit `deploy/amd64/config.toml`, especially `[llm].base_url`.
4. From the repository root:

   ```bash
   docker compose -f deploy/amd64/docker-compose.yml pull
   docker compose -f deploy/amd64/docker-compose.yml up -d
   docker compose -f deploy/amd64/docker-compose.yml logs -f
   ```

Whisper model downloads are persisted in the `whisper-models` named volume.
Stop the stack with:

```bash
docker compose -f deploy/amd64/docker-compose.yml down
```

## Images

GitHub Actions publishes `linux/amd64` images to GHCR with explicit `amd64`
tags: `koko-server:amd64`, `koko-whisper:amd64`, and
`koko-phonemize:amd64`. The workflow builds on GitHub's native AMD64 runner.
