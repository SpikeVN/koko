# Jetson Xavier deployment

This Compose deployment targets a 32 GB Jetson Xavier on JetPack 5.1 or newer
(L4T R35.2.1+). GitHub Actions builds and publishes the ARM64 images; the
Jetson only pulls and runs them. The stack has three services:

- `websocket`: Koko's public websocket/TTS server on port 6942.
- `whisper-live`: GPU ASR on the private Compose network.
- `phonemize`: CPU-only sea-g2p HTTP service on the private Compose network.

The GPU images are based on `jetson-containers` R35 images. These images carry
the JetPack-matched CUDA, CTranslate2, and ONNX Runtime builds; do not replace
their runtime packages with PyPI CUDA wheels.

## Prerequisites

1. Flash JetPack 5.1+ (L4T R35.2.1 or later) and install Docker plus NVIDIA
   Container Runtime.
2. Copy `.env.example` to `.env`. Images are published at
   `ghcr.io/spikevn/koko-*`.
3. Authenticate to GHCR if the package is private.
4. Fetch the required TTS files into `tts_model/` using `tools/fetch_models.py`.
5. Edit `config.toml` in this directory. In particular, set `[llm].base_url` to
   an address reachable from the Compose network.

## Start

From the repository root on the Jetson:

```bash
docker compose -f deploy/jetson/docker-compose.yml pull
docker compose -f deploy/jetson/docker-compose.yml up -d
docker compose -f deploy/jetson/docker-compose.yml logs -f
```

Connect clients to `ws://<jetson-ip>:6942`. Whisper's Hugging Face model cache
is stored in the `whisper-models` named volume, so subsequent starts do not
download it again. Stop the stack with:

```bash
docker compose -f deploy/jetson/docker-compose.yml down
```

## Image builds

The workflow `.github/workflows/build-jetson-images.yml` publishes
`koko-server`, `koko-whisper`, and `koko-phonemize` as `linux/arm64` images to
GHCR using the `jp5` and `latest` tags. Compose pulls the explicit `jp5` tag.
It uses QEMU on GitHub's runner to execute ARM64 build steps. The GPU
images remain tied to the Jetson-containers R35 base images and must match the
host's L4T R35 ABI family.

The phonemizer does not use CUDA. Its Python 3.12 ARM64 base is intentional:
`sea-g2p` requires Python 3.10+, while the JetPack 5 GPU images use Python 3.8.
