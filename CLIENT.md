# CLIENT.md — writing a client for koko

Same protocol regardless of implementation language: one websocket
connection to `ws://<server>:6942` carries both binary audio frames and
JSON text messages. Interleaving is allowed; binary frames are
microphone audio, JSON frames are control.

## The handoff

- **Client → server**: raw mono PCM16 microphone audio (16 kHz, ~20 ms
  per frame). No headers, no encoding envelope — the bytes ARE the audio.
- **Client → server (control)**: `hello` / `asr_language` / `asr_auto_detect` / `target_language` / `tts_voice` / `clear_context` / `bye` JSON frames.
- **Server → client**: TTS output as binary PCM16 frames (48 kHz), each
  preceded by a JSON `{"type": "audio", "rate": <Hz>}` header — **the
  client decodes and plays these locally**. The server does not play
  audio; it only transmits it.

## Full lifecycle

1. Connect.
2. Send `hello`.
3. Stream mic frames as binary; meanwhile listen for events.
4. Pick the `rate` from `audio` headers and play binary frames that
   follow.
5. On quit: send `bye`, drain any remaining audio you want, close.
6. Each connection is an independent session. Multiple clients can stream at
   once and may use different ASR languages and TTS voices.

## Wire protocol v1

### Client → server

| Frame | Format | Meaning |
|---|---|---|
| *binary* | mono PCM16 LE @ 16 kHz | ~20 ms mic chunks, streamed continuously |
| `hello` | `{"type": "hello", "language": "en", "target_language": "English"}` | announces client; `language` sets ASR source language and `target_language` sets LLM output language |
| `asr_language` | `{"type": "asr_language", "language": "vi"}` | switch WhisperLive's language for the active session; it reconnects on the next active audio block |
| `asr_auto_detect` | `{"type": "asr_auto_detect", "enabled": true}` | enable or disable Whisper language detection for the active session; disabled by default |
| `target_language` | `{"type": "target_language", "language": "English"}` | switch the LLM output language and clear translation context |
| `tts_voice` | `{"type": "tts_voice", "voice": "Adam"}` | switch the active VieNeu preset for later speech; the selected name must be in `ready.tts_voices` |
| `clear_context` | `{"type": "clear_context"}` | discard buffered source text and LLM conversation history |
| `bye` | `{"type": "bye"}` | clean disconnect from server side |

### Server → client

| Type | Fields | Meaning |
|---|---|---|
| `ready` | `asr_language`, `asr_auto_detect`, `target_language`, `tts_voices`, `tts_voice` (VieNeu only) | session is live; safe to start streaming. The first ready message lists the active ASR mode and selectable `[name, description]` presets. |
| `tts_voice` | `voice` | confirms that subsequent VieNeu audio uses this preset |
| `asr_language` | `language` | confirms the active WhisperLive language |
| `asr_auto_detect` | `enabled` | confirms Whisper automatic language detection |
| `target_language` | `language` | confirms the active LLM output language |
| `error` | `detail` | a stage crashed or a control request was invalid |
| `audio` | `rate` (e.g. `48000`) | **announcement**: binary frames which follow immediately encoded mono PCM16 |
| `partial` | `text` | temporary ASR guess for the current speech cycle |
| `final` | `text` | confirmed ASR segment (per ~5 s of speech) |
| `speak` | `text` | transcript being sent to the LLM |
| `translation_partial` | `text` | temporary token-level translation preview for the current sentence |
| `translation` | `text` | LLM translation in the active target language |
| `error` | `detail` | any exception message surfaced to the client |

### Playback client-side

- On `audio`: set the player's sample rate to the value in the header,
  then decode the binary PCM16 LE frames that follow into floats
  (`float(buf)/32768`) and write them to the audio output stream in
  real time.
- **Do not** buffer forever — audio arrives as soon as the TTS stage
  has it, which can be mid-generation; the client should stream out
  chunk by chunk like the reference client does.

## Reference implementations to copy from

- **Python**: `clients/client.py` — captures mic via sounddevice, plays TTS
  with sounddevice in a dedicated thread (see its `_play` function for
  the re-open-on-rate-change pattern, and `join()` before kill to avoid
  a portaudio segfault).
- **JavaScript (browser)** — sample skeleton:

  ```js
  const ws = new WebSocket("ws://server:6942");
  ws.onopen = () => ws.send(JSON.stringify({"type":"hello","language":"en"}));

  // SENDING: capture at 16 kHz mono (AudioWorklet recommended; the
  // deprecated ScriptProcessorNode also works). Convert captured
  // float32 [-1, 1] frames to int16 before sending:
  //   const c = Math.round(Math.max(-1, Math.min(1, f)) * 32767);
  //   view.setInt16(offset, c, true);               // little-endian
  ws.send(pcm16Bytes);

  let expectedRate = 48000, ctx = null;
  ws.onmessage = (e) => {
    if (typeof e.data === "string") {                 // control frame
      const m = JSON.parse(e.data);
      console.log(m.type, m.text ?? m.detail ?? "");
      if (m.type === "audio") expectedRate = m.rate;  // next binary chunk
      return;
    }
    // binary audio: decode int16 LE -> float32 with a DataView
    // (x / 32768), then feed into an audio graph scheduled at
    // ctx.currentTime...
  };
  ```

- **Android (Kotlin)** — use `OkHttp` WebSocket + `AudioTrack`:
  - capture: `MediaRecord`/`AudioRecord` at 16 kHz mono PCM16, write
    `ByteArray`s to `ws.send(ByteString)`;
  - playback: on `audio` header read `rate`, create `AudioTrack` with
    that rate, `CHANNEL_OUT_MONO`, `AUDIO_FORMAT_PCM_16_BIT`, then write
    the binary frame payloads as they arrive.
  - Beware throughput vs latency: re-create the `AudioTrack` if `rate`
    changes mid-session (it can — TTS is configured at 48 kHz, but the
    header is authoritative).

## Gotchas

- Connections are independent; do not use duplicate sockets for the same
  capture session unless the application intentionally needs separate contexts.
- Backpressure: inbound frames are silently dropped by the server when
  its TTS output queue is nearly full. Keep buffering shallow (~20 ms
  chunks); extra buffering hurts latency more than it helps.
- Audio and text frames interleave; a `final`/`translation` message can
  arrive while binary audio is mid-flight — handle both on one event loop.
