# koko — live interpretation over websockets

koko is a speech-to-speech **live interpreter**: someone speaks into your
device, ASR transcribes them in real time, an LLM translates the running
transcript into Vietnamese, and TTS speaks the translation — all pipelined
so audio of the first sentence starts while the speaker is still talking.
The server hosts the whole pipeline behind **one websocket port** (6942);
the client just streams microphone audio up and plays TTS audio as it
arrives. **All audio playback happens on the client** — the server sends
synthesized PCM over the socket and never opens an output stream.

## Architecture

Each websocket connection has its own session pipeline, so clients can use
different ASR languages and TTS voices concurrently. Inside each session,
asyncio stages pass events over a private `Bus`; the server shares loaded TTS
model resources:

```
 client mic (PCM16 16 kHz)
        │  binary websocket frames
        ▼
 ┌─────────────┐   SOURCE_CHUNK / SOURCE_SPEAKS
 │   WsFeed    │──────────────►┌─────────────┐
 │ (20 ms→1 s  │               │  ASR stage   │  WhisperLive (only engine)
 │  packing)   │               │             │  → PARTIAL_TEXT / FINAL_TEXT
 └─────────────┘               └──────┬──────┘
                               └──────┬───────┘
                                      │ transcript text
                                      ▼
                            ┌─────────────────────┐
                            │ Interpretation gate │  buffers text; fires a
                             │  (every 8 words,    │  burst to the LLM once
                             │  pause, or 2 s ASR  │  release_words hit, pause
                             │  stall)             │  ends, or ASR stalls
                             └─────────┬───────────┘
                            └─────────┬───────────┘
                                      ▼
                            ┌─────────────────────┐
                            │     LLM stage       │  OpenAI-compatible API
                            │  (streaming)        │  → ASSISTANT_CHUNK
                            └─────────┬───────────┘  per sentence
                                      ▼
                            ┌─────────────────────┐
                            │      TTS stage       │  VieneuLite (ONNX, 48 kHz)
                            │                      │  → audio queue
                            └─────────┬───────────┘
                                      │  {"type":"audio"} + binary PCM16
                                      ▼
                                 client speakers
```

Stages and files:

| component | file | what it does |
|---|---|---|
| ASR | `koko/engine/asr.py` | streaming Whisper via a separate WhisperLive server (`whisper_live_url`) |
| Gate | `koko/engine/gate.py` | decides *when* buffered transcript goes to the LLM |
| LLM | `koko/engine/llm.py` | streaming translation; groups tokens into sentences |
| TTS | `koko/engine/tts.py`, `koko/engine/tts_vieneu.py` | VieNeu-TTS via local ONNX, or null dry-run output |
| phonemes | `koko/engine/phonemize.py` | remote G2P client (text → phonemes over HTTP) |
| server | `koko/server.py` | hosts the whole pipeline on port 6942 |
| reference client | `clients/client.py` | streams mic audio, plays returned TTS |

## Setup

1. **Get the code** and create the environment (needs Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

2. **Fetch model weights** (TTS + whisper weights from the HF hub):

   ```bash
   uv run tools/fetch_models.py   # skip files already present
   ```

   See `MODELS.md` for the full list of what lands where.

3. **Configure** (or accept defaults) — everything lives in one TOML file,
   `config.toml`, loaded at startup by `koko/engine/config.py`. There are no env
   vars. The important sections:

   - `[llm]` `base_url` / `model` — any OpenAI-compatible server
     - `[tts]` `backend` (`vieneu` / `null`), `vieneu_voice` — voice selection. Set `vieneu_voices_path` to use another compatible preset JSON.
   - `[asr]` `whisper_live_url` / `whisper_live_model` / `whisper_live_vad` —
     WhisperLive server endpoint + client handshake facts (only ASR engine);
     `whisper_live_host/_port/_backend/_max_clients/_max_connection_s` —
     how the switchless `koko/whisper_live_server.py` is launched
   - `[phonemize]` `url` — phonemizer HTTP endpoint (box: `tools/phonemize_server.py`)
     - `[gate]` `release_words` / `gap_reset_s` / `no_new_words_s` — when translation fires

   Run the server with a per-machine config when URLs differ:

   ```bash
    uv run koko-server --config /path/to/machine.toml
   ```

4. **Check the machine** has the weights / GPU / services:

   ```bash
   uv run tools/check_laptop.py     # optional sanity script
   uv run tools/setup_laptop.py     # optional full setup
   ```

5. **Run the WhisperLive ASR server** (the Koko server automatically starts a
   missing local WhisperLive service; this standalone command is useful for
   running it separately):
   WhisperLive lives in its own `.venv-whisperlive` because it clashes with
   the TTS's `onnxruntime-gpu`.

   ```bash
   uv run whisper_live.py         # ASR server: ws://<this-host>:9090, switchless
   ```

   Host, port, model, backend, max clients and connection cap come from the
   `[asr]` section of `config.toml`.

6. **Run:**

   ```bash
   uv run koko-server              # starts local phonemizer + WhisperLive as needed
   uv run koko-client              # Tkinter client: mic → server → speakers
   ```

### Jetson Xavier Docker deployment

For a JetPack 5.1+ (L4T R35.2.1+) Xavier deployment, Compose definitions for
the websocket server, GPU WhisperLive server, and phonemizer are in
[`deploy/jetson/`](deploy/jetson/README.md). They use Jetson-containers R35
images and publish only the websocket port.

## Quick facts

- Input to the server: raw mono **PCM16 @ 16 kHz** in ~20 ms binary
  frames + JSON control messages (`hello`, `bye`).
- Output to the client: JSON `{"type":"audio","rate":<Hz>}` before each
  binary **PCM16 @ 48 kHz** chunk, plus live text events. Playback is the
  client's job.
- Full wire protocol, including how to build a client in any language:
  see **CLIENT.md**.

## Development

`tools/bench_*.py` scripts benchmark LLM / TTS / audio paths.
`config.toml` (loaded by `koko/engine/config.py`) is the single source of truth
for external facts (URLs, model names); it's validated at startup — unknown
sections/keys and wrong value types are rejected, and unknown ASR / TTS
backend values raise `ValueError`.
