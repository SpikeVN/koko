# koko — agent notes

Live speech-to-speech interpreter. One websocket server (`ws_server.py`,
port 6942) hosts the full pipeline (ASR → gate → streaming LLM translation →
TTS); clients stream mic PCM up and play returned TTS audio. **Playback is
client-side only** — the server never opens an audio output stream; the
TTS stage's `_Sink` in `_session()` just queues numpy arrays that get
serialized to binary websocket frames.

## Layout

- `engine/config.py` — single source of truth for all external facts
  (URLs, ports, model names, sample rates). `KOKO_*` env vars override
  (see `apply_env_overrides`). Do not hardcode those facts anywhere else.
- `engine/bus.py` / `engine/events.py` — pub/sub event bus stages run on.
  Kinds: `SOURCE_CHUNK`, `SOURCE_SPEAKS`, `PARTIAL_TEXT`, `FINAL_TEXT`,
  `SPEAK`, `ASSISTANT_CHUNK`.
- `engine/asr.py` — faster-whisper (in-process, default) or WhisperLive.
- `engine/gate.py` — buffers transcript, releases to LLM after
  `interpretation_delay_s` (5 s) of continuous speech; a `gap_reset_s`
  (1.2 s) silence gap restarts the cycle.
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

1. **Backend/ASR validation**: `WsServer._load_models` raises
   `ValueError` for unknown `cfg.tts.backend` or `cfg.asr.engine` values.
   Keep it that way — the old `else NullTts()`-style silent fallbacks
   masked config typos (`vietneu` vs `vieneu` cost real debugging time).
2. **Wire format**: client→server binary frames are mono PCM16 @ 16 kHz,
   ~20 ms each; server→client audio is a JSON header
   `{"type":"audio","rate":<Hz>}` followed by one binary PCM16 frame —
   every chunk, not just the first.
3. **Silence gating**: `WsFeed` packs client frames into 1 s blocks and
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

- Run server: `uv run ws_server.py`; client: `uv run ws_client.py
  [ws://host:6942] [--language en] [--device <name>]`.
- Weights land via `tools/fetch_models.sh` (see `MODELS.md`); they are
  not in git and never should be.
- Voice: `KOKO_VIENEU_VOICE` or `tts.vieneu_voice` in config; names come
  from `tts_model/voices_v3_turbo.json` `presets` (exact, accented).
- Benchmarks: `tools/bench_*.py`.
