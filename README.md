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

One websocket connection per session (only one client at a time). Inside
the server, the pipeline is a set of asyncio stages passing events over a
shared `Bus`:

```
 client mic (PCM16 16 kHz)
        │  binary websocket frames
        ▼
 ┌─────────────┐   SOURCE_CHUNK / SOURCE_SPEAKS
 │   WsFeed    │──────────────►┌─────────────┐
 │ (20 ms→1 s  │               │  ASR stage   │  faster-whisper (in-process)
 │  packing)   │               │  or Whisper- │  → PARTIAL_TEXT / FINAL_TEXT
 └─────────────┘               │   Live       │
                               └──────┬───────┘
                                      │ transcript text
                                      ▼
                            ┌─────────────────────┐
                            │ Interpretation gate │  buffers text; fires the
                            │  (~5 s speech, 1.2 s│  LLM turn after 5 s of
                            │  gap resets cycle)  │  continuous speech
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
| ASR | `engine/asr.py` | streaming Whisper (in-process faster-whisper by default, or a WhisperLive server) |
| Gate | `engine/gate.py` | decides *when* buffered transcript goes to the LLM |
| LLM | `engine/llm.py` | streaming translation; groups tokens into sentences |
| TTS | `engine/tts.py`, `engine/tts_vieneu.py` | VieNeu-TTS v3 Turbo via local ONNX graphs (torch-free) |
| phonemes | `engine/phonemize.py` | remote G2P client (text → phonemes over HTTP) |
| server | `ws_server.py` | hosts the whole pipeline on port 6942 |
| reference client | `ws_client.py` | streams mic audio, plays returned TTS |

## Setup

1. **Get the code** and create the environment (needs Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

2. **Fetch model weights** (TTS + whisper weights from the HF hub):

   ```bash
   tools/fetch_models.sh          # skip files already present
   ```

   See `MODELS.md` for the full list of what lands where.

3. **Configure** (or accept defaults) via env vars — everything lives in
   `engine/config.py`, and `KOKO_*` env vars override it. The important
   ones:

   - `KOKO_LLM_BASE_URL` / `KOKO_LLM_MODEL` — any OpenAI-compatible server
   - `KOKO_TTS` (`vieneu` / `null`), `KOKO_VIENEU_VOICE` — voice selection
   - `KOKO_ASR_ENGINE` (`faster_whisper` default, or `whisper_live`)
   - `KOKO_PHONEMIZE_URL` — phonemizer HTTP endpoint (box: `tools/phonemize_server.py`)

4. **Check the machine** has the weights / GPU / services:

   ```bash
   uv run tools/check_laptop.py     # optional sanity script
   uv run tools/setup_laptop.sh     # optional full setup
   ```

5. **Run:**

   ```bash
   uv run ws_server.py            # server: :6942
   uv run ws_client.py            # client: mic → server → speakers
   ```

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
`engine/config.py` is the single source of truth for external facts
(URLs, model names); ASR and TTS backend values are validated at startup —
unknown values raise `ValueError`.
