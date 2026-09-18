# Jetson Xavier deployment

This Compose deployment targets a 32 GB Jetson Xavier on JetPack 5.1 or newer
(L4T R35.2.1+). It builds ARM64 images on the Jetson and runs three services:

- `websocket`: Koko's public websocket/TTS server on port 6942.
- `whisper-live`: GPU ASR on the private Compose network.
- `phonemize`: CPU-only sea-g2p HTTP service on the private Compose network.

The GPU images are based on `jetson-containers` R35 images. These images carry
the JetPack-matched CUDA, CTranslate2, and ONNX Runtime builds; do not replace
their runtime packages with PyPI CUDA wheels.

## Prerequisites

1. Flash JetPack 5.1+ (L4T R35.2.1 or later) and install Docker plus NVIDIA
   Container Runtime.
2. Install [jetson-containers](https://github.com/dusty-nv/jetson-containers)
   on the Xavier. It configures Docker to use the Jetson GPU runtime.
3. Fetch the required TTS files into `tts_model/` using `tools/fetch_models.py`.
4. Edit `config.toml` in this directory. In particular, set `[llm].base_url` to
   an address reachable from the Compose network.

## Start

From the repository root on the Jetson:

```bash
docker compose -f deploy/jetson/docker-compose.yml build
docker compose -f deploy/jetson/docker-compose.yml up -d
docker compose -f deploy/jetson/docker-compose.yml logs -f
```

Connect clients to `ws://<jetson-ip>:6942`. Whisper's Hugging Face model cache
is stored in the `whisper-models` named volume, so subsequent starts do not
download it again. Stop the stack with:

```bash
docker compose -f deploy/jetson/docker-compose.yml down
```

## Image selection

The defaults use maintained R35 tags and are compatible across the R35 JetPack
5 series. If your installed jetson-containers checkout provides locally built
images instead, copy `.env.example` to `.env` and set `ONNXRUNTIME_IMAGE` and
`WHISPER_IMAGE` to those image names before building. Keep both images on the
same L4T R35 ABI family as the host.

The phonemizer does not use CUDA. Its Python 3.12 ARM64 base is intentional:
`sea-g2p` requires Python 3.10+, while the JetPack 5 GPU images use Python 3.8.
