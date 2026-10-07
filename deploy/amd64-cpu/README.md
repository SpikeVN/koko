# AMD64 CPU-only deployment

This deployment runs the complete stack on Linux AMD64 without an NVIDIA GPU,
CUDA, or NVIDIA Container Toolkit. TTS uses CPU ONNX Runtime and ASR uses
CPU Faster-Whisper. CPU inference is slower than the GPU deployments; the
default Whisper model may require substantial RAM and can increase latency.

## Start

1. Install Docker with the Compose plugin.
2. Fetch the TTS files into `tts_model/` with `tools/fetch_models.py`.
3. Edit `deploy/amd64-cpu/config.toml`, especially `[llm].base_url`.
4. From the repository root:

   ```bash
   docker compose -f deploy/amd64-cpu/docker-compose.yml pull
   docker compose -f deploy/amd64-cpu/docker-compose.yml up -d
   docker compose -f deploy/amd64-cpu/docker-compose.yml logs -f
   ```

Whisper model downloads are persisted in the `whisper-models` named volume.
Stop the stack with:

```bash
docker compose -f deploy/amd64-cpu/docker-compose.yml down
```

GitHub Actions publishes `linux/amd64` CPU images with the `amd64-cpu` tag.
