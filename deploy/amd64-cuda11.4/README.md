# AMD64 CUDA 11.4 NVIDIA GPU deployment

This Compose deployment targets Linux AMD64 hosts with an NVIDIA GPU, a CUDA
11.4-compatible driver, Docker, and NVIDIA Container Toolkit. It runs the same
three-service stack as `deploy/amd64/`, but packages CUDA 11.4 and cuDNN 8 in
the GPU containers.

- `koko-server`: public websocket and Vieneu TTS service on port 6942.
- `koko-whisper`: CUDA Faster-Whisper ASR service on the private Compose network.
- `koko-phonemize`: CPU-only sea-g2p service on the private Compose network.

GitHub Actions publishes explicit `amd64-cuda11.4` GHCR tags. This keeps the
CUDA 11.4 dependency explicit. `koko-whisper` pins CTranslate2 3.24.0, the
final release with CUDA 11/cuDNN 8 support, with Faster-Whisper 0.10.1 as its
matching CUDA 11/Python 3.8 release. `koko-server` uses ONNX Runtime 1.16.3,
the corresponding CUDA 11/cuDNN 8 release.

## Start

1. Install Docker and NVIDIA Container Toolkit, then verify GPU access:

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04 nvidia-smi
   ```

2. Fetch the required TTS files into `tts_model/` with `tools/fetch_models.py`.
3. Edit `deploy/amd64-cuda11.4/config.toml`, especially `[llm].base_url`.
4. From the repository root, pull and start the stack:

   ```bash
   docker compose -f deploy/amd64-cuda11.4/docker-compose.yml pull
   docker compose -f deploy/amd64-cuda11.4/docker-compose.yml up -d
   docker compose -f deploy/amd64-cuda11.4/docker-compose.yml logs -f
   ```

Whisper model downloads are persisted in the `whisper-models` named volume.
Stop the stack with:

```bash
docker compose -f deploy/amd64-cuda11.4/docker-compose.yml down
```
