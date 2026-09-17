# koko — agent notes

Live speech-to-speech interpreter. One websocket server (`ws_server.py`,
port 6942) hosts the full pipeline (ASR → gate → streaming LLM translation →
TTS); clients stream mic PCM up and play returned TTS audio. **Playback is
client-side only** — the server never opens an audio output stream; the
TTS stage's `_Sink` in `_session()` just queues numpy arrays that get
serialized to binary websocket frames.

## Layout

- `engine/config.py` — single source of truth for all external facts
  (URLs, ports, model names, sample rates); values are loaded from
  `config.toml` via `load_config(path)` (stdlib `tomllib`). No env vars.
  Unknown keys / sections and wrong types raise at load; per-machine files
  are passed with `ws_server.py --config PATH`. Do not hardcode those facts
  anywhere else.
- `engine/bus.py` / `engine/events.py` — pub/sub event bus stages run on.
  Kinds: `SOURCE_CHUNK`, `SOURCE_SPEAKS`, `PARTIAL_TEXT`, `FINAL_TEXT`,
  `SPEAK`, `ASSISTANT_CHUNK`.
- `engine/asr.py` — WhisperLive (the only ASR engine; needs a
  `whisper_live_url` server). It speaks the whisper-live 0.10 handshake
  (JSON config → `SERVER_READY`, then rolling `segments` updates that are
  deduped); every ASR fact — client and server — lives in `config.toml`
  `[asr]`. The server process (`whisper_live_server.py`) is switchless.
- `engine/gate.py` — buffers transcript; releases to LLM once
  `release_words` (5) source words accumulate, or early when a
  `gap_reset_s` (1.2 s) silence gap ends the utterance with text buffered.
- `engine/llm.py` — OpenAI-compatible streaming; emits sentence-sized
  `ASSISTANT_CHUNK`s so TTS starts before generation finishes.
- `engine/tts.py`, `engine/tts_vieneu.py` — VieneuLite ONNX pipeline
  (torch-free); phonemes come from the remote phonemizer.
- `engine/phonemize.py` — HTTP client for the G2P endpoint
  (`tools/phonemize_server.py` on the LLM host; sea-g2p is Rust-only on
  the server side).
- `engine/pipeline.py`, `engine/source.py`, `engine/tts.py`'s
  `AudioPlayer` — the local (non-websocket) pipeline; `source.py` = mic,
  `AudioPlayer` = local speakers. Used for benching / offline testing.
- `ws_server.py` — the thing you actually run; protocol documented in
  `CLIENT.md` and the module docstring.
- `ws_client.py` — reference client (mic → socket → playback).
- `tools/` — model fetch, phonemize server, benches.

## Must-know invariants

1. **Backend validation**: ASR is unconditionally WhisperLive (the
   faster-whisper engine is gone). `WsServer._load_models` still raises
   `ValueError` for unknown `cfg.tts.backend` values — keep it that way;
   the old `else NullTts()`-style silent fallbacks masked config typos
   (`vietneu` vs `vieneu` cost real debugging time).
2. **Wire format**: client→server binary frames are mono PCM16 @ 16 kHz,
   ~20 ms each; server→client audio is a JSON header
   `{"type":"audio","rate":<Hz>}` followed by one binary PCM16 frame —
   every chunk, not just the first.
3. **Silence gating**: `WsFeed` packs client frames into 250 ms blocks and
   drops blocks below −45 dBFS before they reach whisper (silence makes
   whisper hallucinate); it always emits `SOURCE_SPEAKS` for the gate.
4. Frame backpressure: inbound PCM is dropped once `out_q` is near full
   (`ws_server.py` `_session`); the TTS output queue similarly drops.
   Latency > throughput wins here — don't add unbounded buffering.
5. Session teardown bounds every cleanup with `asyncio.wait(..., timeout)`
   because a wedged stage (LLM call, executor thread) must not keep the
   one-connection-at-a-time `busy` lock.
6. `ws_client.py`'s `_play` thread must be `join()`ed after `stop_evt`
   (never killed mid-`stream.write`) — portaudio segfaults otherwise.

## Ops

- Run server: `uv run ws_server.py [--config PATH]`; client: `uv run
  ws_client.py [ws://host:6942] [--language en] [--device <name>]`.
- Config: `config.toml` at repo root, loaded at startup; per-machine URLs
  via a different file and `--config PATH`. There are no env vars.
- Weights land via `tools/fetch_models.sh` (see `MODELS.md`); they are
  not in git and never should be.
- Voice: `tts.vieneu_voice` in `config.toml`; names come from
  `tts_model/voices_v3_turbo.json` `presets` (exact, accented).
- Benchmarks: `tools/bench_*.py`.
