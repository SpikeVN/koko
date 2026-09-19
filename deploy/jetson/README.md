# Jetson Xavier deployment

This Compose deployment targets a 32 GB Jetson Xavier on JetPack 5.1 or newer
(L4T R35.2.1+). The stack has three services:

- `koko-server`: Koko's public websocket/TTS server on port 6942.
- `koko-whisper`: GPU ASR on the private Compose network.
- `koko-phonemize`: CPU-only sea-g2p HTTP service on the private Compose network.

`koko-whisper` is built from `ghcr.io/spikevn/faster-whisper:jp5`, which has
CTranslate2 3.24.0 compiled for Xavier's `sm_72` GPU and Faster-Whisper 1.2.0
built from source for JetPack 5 Python 3.8. Do not replace its runtime packages
with PyPI CUDA wheels.

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

GitHub Actions publishes the ARM64 `koko-server:jp5` and
`koko-phonemize:jp5` images from
`.github/workflows/build-jetson-images.yml`. The CUDA Whisper image is not
built in GitHub Actions; build it directly on the Xavier as described below.

The reusable `faster-whisper:jp5` base is built in the Jetson-containers
repository. Rebuild and publish it only when native CTranslate2 or
Faster-Whisper recipes change; Koko changes rebuild only `koko-whisper`.

Build `koko-whisper` directly on the Xavier. If the base image is private,
authenticate to GHCR first:

```bash
docker login ghcr.io
docker build --pull \
  --tag ghcr.io/spikevn/koko-whisper:jp5 \
  --file deploy/jetson/whisper-live.Dockerfile .
docker compose -f deploy/jetson/docker-compose.yml up -d --no-deps --force-recreate koko-whisper
```

Confirm that the local image uses the Xavier GPU:

```bash
docker compose -f deploy/jetson/docker-compose.yml exec koko-whisper \
  python3 -c 'import ctranslate2; assert ctranslate2.get_cuda_device_count() == 1'
```

The phonemizer does not use CUDA. Its Python 3.12 ARM64 base is intentional:
`sea-g2p` requires Python 3.10+, while the JetPack 5 GPU images use Python 3.8.
