# koko — real-time spoken interpreter

Audio in → live speech-to-text → LLM interpreter (Qwen, local) → live text-to-speech out.
The TTS starts speaking after the source speaker has been talking continuously for
`KOKO_DELAY_S` seconds (standard interpretation style).

```
mic ─► SOURCE_CHUNK/SOURCE_SPEAKS ─► ASR ─► FINAL_TEXT ─► gate ─► SPEAK ─► LLM ─► ASSISTANT_CHUNK ─► TTS ─► speaker
                          20ms blocks        1s chunks    buffer     5s cycle  stream      sentence queue
```

## Architecture

`engine/` — every stage is a queue consumer behind one interface, so each can be
swapped independently:

| file | role |
|---|---|
| `engine/config.py` | all URLs, ports, model names, thresholds (env-overridable, `KOKO_*`) |
| `engine/events.py` | typed events crossing stage boundaries |
| `engine/bus.py` | event bus (per-subscriber queues; thread-safe put + asyncio pump) |
| `engine/source.py` | mic capture + energy VAD → `SOURCE_CHUNK` / `SOURCE_SPEAKS` |
| `engine/asr.py` | streaming transcription: `FasterWhisperAsr` (default) or `WhisperLiveAsr` |
| `engine/gate.py` | **5s interpretation trigger** — accumulates ASR text, releases to LLM on the speech timer |
| `engine/llm.py` | OpenAI-compatible streaming client, token → sentence chunking |
| `engine/tts.py` | TTS backends (`NullTts` for dry runs, `VieneuTts`) + audio player queue |
| `engine/pipeline.py` | wires everything, owns lifecycle & shutdown |
| `engine/monitor.py` | latency percentiles per event |

`main.py` is the entrypoint.

## External services & ports (dev box)

| service | URL | what runs there | quant / VRAM guide |
|---|---|---|---|
| LLM (vLLM, OpenAI-compatible) | `http://127.0.0.1:8000/v1` | Qwen 9B-class chat model | bf16 if ≥ 24 GB free; fp8 (Ada/Hopper) ≈ 10 GB; 4-bit AWQ/GGUF if < 10 GB |
| WhisperLive (optional ASR server) | `ws://127.0.0.1:9090` | faster-whisper backend | prefer `large-v3-turbo` int8_float16 (~2 GB); `distil-large-v3` fp16 ≈ 2 GB |
| diart / pyannote (speaker labels) | in-process, no port | downloaded from HF on first run | fp16 |
| vienneu-tts | in-process via `VieneuTts` | TTS engine | vendor's own cfg; runs offline after model download |

GPU budget: run Whisper + Qwen diart TTS on one 24 GB card comfortably. Whisper and diart are light
enough to share; keep the LLM resident in vLLM with `--gpu-memory-utilization 0.35` if co-located
with the ASR. **Do not let TTS output feed the mic** — headphones, or a separate output device per
`KOKO_TTS_SR`.

## Run

```bash
uv run python main.py                    # default config
KOKO_DELAY_S=2 KOKO_TTS=null uv run python main.py        # faster speech cycle, no audio
KOKO_ASR_ENGINE=whisper_live KOKO_WHISPER_LIVE_URL=ws://127.0.0.1:9090 uv run python main.py
```

## Environment knobs (subset)

| var | default | meaning |
|---|---|---|
| `KOKO_DELAY_S` | `5` | interpretation start delay (s) |
| `KOKO_ASR_ENGINE` | `faster_whisper` | or `whisper_live` |
| `KOKO_ASR_MODEL` | `Systran/faster-whisper-large-v3-turbo` | faster-whisper model |
| `KOKO_LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible endpoint |
| `KOKO_LLM_MODEL` | `Qwen/Qwen3-8B-AWQ` | chat model served there |
| `KOKO_TTS` | `null` | `null` (dry run) or `vieneu` |
| `KOKO_LANGUAGE` | `en` | source speech language |

## Production target: NVIDIA Jetson AGX Xavier 32 GB (CUDA 11.4)

Constraints that shape this box: JetPack 5 / CUDA 11.4 / Volta (`sm_72`) / 64-bit LPDDR4x
(~136 GB/s — the real inference bottleneck) / no official CUDA 12 stack for `sm_72`.

### Deployment: containerized, not bare-metal

Building llama.cpp (and getting a sane torch stack) on Jetson is a real hassle, so the
production plan runs **inside a container image that ships a working torch + vLLM +
llama.cpp out of the box**. Consequences for the README advice below:

- The "CUDA 11.4 makes vLLM impossible" reasoning above only applied to bare-metal pip
  installs; inside a container with bundled CUDA/torch wheels that constraint is
  lifted, and **vLLM becomes viable on Xavier** if the image's build supports `sm_72`
  (verify with `python -c "import torch; print(torch.cuda.get_arch_list())"` → must
  contain `7.2`).
- Whatever server the container brings (vLLM or llama.cpp) speaks the OpenAI-compatible
  protocol, so our code is unchanged — only `KOKO_LLM_BASE_URL` / `KOKO_LLM_MODEL`
  flip depending on which one you run.

### Plan of record

| stage | run as | specifics |
|---|---|---|
| LLM | **inside the container**: `llama-server :8000` (GGUF Q4_K_M ≈ 6–7 GB) or vLLM (AWQ/fp8) | both expose `/v1/chat/completions`; expect ~10–20 tok/s gen on Xavier's ~136 GB/s bus either way. Prefer llama.cpp if you want KV-cache-quant / flash-attn knobs; vLLM if batching multiple streams matters. |
| ASR | faster-whisper, `compute_type=int8` | if the container brings a decent torch, try GPU-first, but Xavier CPU handles distil/turbo-int8 models fine in real time — avoid GPU contention with the LLM if vLLM is resident. |
| VAD | RMS energy (`SOURCE_SPEAKS`) | diart/pyannote only if the container's torch actually supports `7.2`/sm_72 — otherwise skip it, the RMS VAD already drives the gate. |
| TTS | in-process (vieneu) or CPU | keep it light; don't let it steal ASR CPU while speech is streaming. |

**Toolchain facts:** JetPack 5 = CUDA 11.4, cuDNN 8, TensorRT 8.5. Container escape of the
bare-metal CUDA ceiling is exactly why a Docker image "just works" where pip installs
would fight on this hardware.

## Interpretation semantics

- The gate tracks speech activity. Continuous speech for `KOKO_DELAY_S` (default: 5s)
  releases the buffer to the LLM; a >1.2s pause resets the timer for a fresh cycle.
- While that turn is in-flight, the user can keep speaking: new text accumulates in
  a separate turn. Barge-in is treated as a new cycle rather than pre-empting the
  TTS (interpretation, not dialogue); revisit in `engine/gate.py`.
- The LLM streams tokens; token buffers are flushed to TTS at sentence boundaries
  so audio starts early.

## Roadmap / not live yet

- vienneu-tts actual engine binding (imports `vieneu` at first use; plug in `engine/tts.py`).
- Real barge-in cancellation of an in-flight LLM stream.
- WhisperLive client is implemented against a single-connection model of the WhisperLive
  protocol; may need adjustments for the exact server version you run.
- Speaker labelling via diart (repo had `voice_seperate.py`; now the VAD signals
  SOURCE_SPEAKS come from simple RMS energy, which ignores noise from other speakers).
