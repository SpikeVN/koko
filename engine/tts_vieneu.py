"""VieNeu-TTS v3 Turbo — torch-free ONNX inference engine (Python 3.8 port).

Ported from vieneu._v3_turbo_engine.onnx_runtime_lite (Apache-2.0), trimmed to
what koko needs:
  * local model directory (tts_model/) instead of huggingface_hub downloads
  * phonemization is *not* performed here — callers pass pre-phonemized text
    (see engine/phonemize.py; sea-g2p's Rust core is py3.10-only)
  * preset voices only (voice-cloning path dropped: no speaker-
    encoder/denoiser wiring on the Jetson yet)

Pipeline (all ONNX + numpy, no torch):
  phoneme text -> tokenizer -> embeddings+speaker anchor -> prefill graph
  -> per-frame decode-step graph + acoustic-cached graph (16 RVQ codebooks,
  external sampling loop in numpy) -> MOSS codec decode_step -> 48 kHz wav

Model dir layout expected:
  tts_model/
    onnx_int8/{vieneu_prefill,vieneu_decode_step,vieneu_acoustic_cached}.onnx
              vieneu_backbone_shared.data  vieneu_v3_heads.npz
              config.json  tokenizer.json
    codec/{moss_audio_tokenizer_decode_full,decode_step,encode}.onnx
          moss_audio_tokenizer_decode_shared.data  codec_browser_onnx_meta.json
    voices_v3_turbo.json
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
from collections import Counter, deque

log = logging.getLogger("koko.tts_vieneu")

DEFAULT_REP_WINDOW = 64        # ~2.5 s audio at 25 frames/s: catches local loops only
_STREAM_LEADIN_FRAMES = 4

ORT = None  # lazy: imported in VieneuLite.__init__ so --help style imports stay fast


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class _ChannelWindow:
    """Sliding-window repetition history for one RVQ codebook (set-like view)."""

    __slots__ = ("_seen", "_order", "_window")

    def __init__(self, window: int):
        self._seen = Counter()
        self._order = deque()
        self._window = int(window)

    def add(self, code: int) -> None:
        self._seen[code] += 1
        if self._window > 0:
            self._order.append(code)
            if len(self._order) > self._window:
                old = self._order.popleft()
                if self._seen[old] <= 1:
                    del self._seen[old]
                else:
                    self._seen[old] -= 1

    def __iter__(self):
        return iter(self._seen)

    def __len__(self):
        return len(self._seen)

    def __bool__(self):
        return bool(self._seen)


class RepetitionHistory:
    """Per-codebook sliding-window history: hist[ch] views channel ch."""

    __slots__ = ("channels",)

    def __init__(self, n_channels: int, window: int = DEFAULT_REP_WINDOW):
        self.channels = [_ChannelWindow(window) for _ in range(n_channels)]

    def __getitem__(self, ch: int) -> _ChannelWindow:
        return self.channels[ch]


class VieneuLite:
    """The v3 Turbo TTS as a manual ONNX pipeline. All-public API + internals
    kept close to upstream OnnxV3LiteEngine for future diffs."""

    SAMPLE_RATE = 48_000

    def __init__(self, model_dir: str, voice: str = "", threads: int = 0):
        global ORT
        import onnxruntime as ort  # noqa: N813
        from tokenizers import Tokenizer
        ORT = ort

        base = Path(model_dir).expanduser()
        vd = base / "onnx_int8"
        cd = base / "codec"

        self._lock = threading.RLock()
        self.base_dir = base

        # -- config ---------------------------------------------------------
        c = json.loads((vd / "config.json").read_text(encoding="utf-8"))
        self.cfg = c
        self.n_vq = int(c["n_vq"])
        self.hidden = int(c["hidden_size"])
        self.L = int(c["num_hidden_layers"])
        self.L_loc = int(c.get("local_num_hidden_layers", 1))
        self.nH_loc = int(c.get("local_num_attention_heads", 8))
        self.hd_loc = self.hidden // self.nH_loc
        self.audio_pad = int(c["audio_pad_token_id"])
        self.tps = int(c["text_prompt_start_token_id"])
        self.tpe = int(c["text_prompt_end_token_id"])
        self.sgs = int(c["speech_generation_start_token_id"])
        self.eos_speech = int(c["speech_generation_end_token_id"])
        self.ref_slot = int(c["audio_ref_slot_token_id"])
        self.use_speaker_embedding = bool(c.get("use_speaker_embedding", False))

        # -- tied embeddings/heads + speaker projection (numpy) --------------
        z = np.load(vd / "vieneu_v3_heads.npz")
        self.text_emb = np.asarray(z["text_emb"], dtype=np.float32)   # (Vt, H)
        self.audio_emb = np.asarray(z["audio_emb"], dtype=np.float32)  # (n_vq, Va, H)
        self.xvec_w = self.xvec_b = self.xvec_ln_w = self.xvec_ln_b = None
        self.xvec_ln_eps = None
        if self.use_speaker_embedding and "xvec_w" in z.files:
            self.xvec_w = np.asarray(z["xvec_w"], dtype=np.float32)    # (H, spk_dim)
            self.xvec_b = np.asarray(z["xvec_b"], dtype=np.float32)
            self.xvec_ln_w = np.asarray(z["xvec_ln_w"], dtype=np.float32)
            self.xvec_ln_b = np.asarray(z["xvec_ln_b"], dtype=np.float32)
            self.xvec_ln_eps = float(z["xvec_ln_eps"])

        # -- phoneme tokenizer (tokenizers lib, torch-free) ------------------
        self.tokenizer = Tokenizer.from_file(str(vd / "tokenizer.json"))

        # -- ONNX sessions ----------------------------------------------------
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads and threads > 0:
            intra = int(threads)
        else:
            intra = min(max((os_cpu() or 8) // 2, 1), 8)
        so.intra_op_num_threads = intra
        self.ort_intra_op_threads = intra
        prov = ["CPUExecutionProvider"]

        def _sess(path):
            log.debug("loading ONNX %s", path.name)
            return ort.InferenceSession(str(path), so, providers=prov)

        self.sess_pre = _sess(vd / "vieneu_prefill.onnx")
        self.sess_dec = _sess(vd / "vieneu_decode_step.onnx")
        self.sess_ac = _sess(vd / "vieneu_acoustic_cached.onnx")
        self.sess_codec_dec = _sess(cd / "moss_audio_tokenizer_decode_full.onnx")

        # streaming codec decoder (decode_step) for streaming synth
        self.sess_codec_step = None
        self._codec_stream_spec = None
        try:
            meta = json.loads((cd / "codec_browser_onnx_meta.json").read_text(encoding="utf-8"))
            self._codec_stream_spec = meta.get("streaming_decode")
            self.sess_codec_step = _sess(cd / "moss_audio_tokenizer_decode_step.onnx")
            self._codec_step_out_names = [o.name for o in self.sess_codec_step.get_outputs()]
        except Exception:
            self.sess_codec_step = None
            log.debug("streaming codec unavailable, infer_stream will fall back")

        # -- preset voices ----------------------------------------------------
        voices_p = base / "voices_v3_turbo.json"
        self.presets = {}
        self.default_voice = None
        if voices_p.is_file():
            d = json.loads(voices_p.read_text(encoding="utf-8"))
            self.presets = d.get("presets", {})
            self.default_voice = d.get("default_voice")
        voice = voice or self.default_voice
        self.voice = None
        if voice:
            self.set_voice(voice)
        log.info(
            "VieNeu v3 Turbo loaded (%s): L=%d L_loc=%d hidden=%d n_vq=%d, "
            "ort_threads=%d, voice=%s",
            vd, self.L, self.L_loc, self.hidden, self.n_vq, intra, voice,
        )

    # -- voices --------------------------------------------------------------
    def list_preset_voices(self):
        return [(k, v.get("description", "")) for k, v in self.presets.items()]

    def set_voice(self, name: str) -> None:
        if name not in self.presets:
            raise ValueError("unknown voice %r; known: %s" % (name, list(self.presets)))
        p = self.presets[name]
        spk = np.asarray(p["speaker_emb"], dtype=np.float32)
        codes = np.asarray(p["codes"], dtype=np.int64) if p.get("codes") else None
        self.voice = (spk, codes)

    # -- numpy helpers -------------------------------------------------------
    def _speaker_anchor(self, speaker_emb):
        if not self.use_speaker_embedding:
            return None
        if self.xvec_w is None:
            raise RuntimeError("heads.npz has no xvec_proj weights")
        v = np.asarray(speaker_emb, dtype=np.float32).reshape(-1)
        if not v.any():
            raise ValueError("speaker_emb is all-zero")
        v = v @ self.xvec_w.T + self.xvec_b
        v = (v - v.mean()) / np.sqrt(v.var() + self.xvec_ln_eps)
        return (v * self.xvec_ln_w + self.xvec_ln_b).astype(np.float32)

    def _embed_rows(self, rows: np.ndarray, anchor=None) -> np.ndarray:
        """rows: (T, n_vq+1) int -> (1, T, H) float32."""
        emb = self.text_emb[rows[:, 0]]
        for ch in range(self.n_vq):
            ids = rows[:, ch + 1]
            valid = ids != self.audio_pad
            safe = np.where(valid, ids, 0)
            emb = emb + self.audio_emb[ch][safe] * valid[:, None]
        if anchor is not None:
            emb = emb + anchor[None]
        return emb[None].astype(np.float32)

    def _sample(self, logits, temperature, top_k, top_p, rep_pen, prev):
        logits = logits.astype(np.float32)
        if not math.isclose(rep_pen, 1.0) and prev:
            idx = np.fromiter(prev, dtype=np.int64, count=len(prev))
            sel = logits[idx]
            logits = logits.copy()
            logits[idx] = np.where(sel < 0, sel * rep_pen, sel / rep_pen)
        if not (temperature and temperature > 0):
            return int(logits.argmax())
        logits = logits / temperature
        V = logits.shape[-1]
        # top_k FIRST so the sort/softmax below runs over k candidates only.
        if top_k and 0 < int(top_k) < V:
            k = int(top_k)
            cand = np.argpartition(logits, -k)[-k:]
        else:
            cand = np.arange(V)
        cs = logits[cand]
        order = np.argsort(cs)[::-1]
        cand = cand[order]
        p = _softmax(cs[order])
        if top_p and top_p < 1.0:
            keep = (np.cumsum(p) - p) < top_p
            p = p * keep
            p = p / p.sum()
        return int(cand[np.random.choice(cand.shape[-1], p=p)])

    # -- prompt build ----------------------------------------------------------
    def _build_rows(self, phonemes: str, ref_codes=None) -> np.ndarray:
        phone_ids = self.tokenizer.encode(phonemes, add_special_tokens=False).ids
        style_id = int(self.cfg.get("default_style_token_id", 16))  # natural
        text_ids = [style_id, self.tps] + list(phone_ids) + [self.tpe]
        T = len(text_ids)
        rows = np.full((T, self.n_vq + 1), self.audio_pad, dtype=np.int64)
        rows[:, 0] = text_ids
        if ref_codes is None:
            return rows
        rc = np.asarray(ref_codes, dtype=np.int64)
        ref = np.full((rc.shape[0], self.n_vq + 1), self.audio_pad, dtype=np.int64)
        ref[:, 0] = self.ref_slot
        ref[:, 1:] = rc
        return np.concatenate([rows, ref], axis=0)

    # -- acoustic frame: cached local-layer steps + numpy heads/sampling --------
    def _empty_past(self):
        empty = np.zeros((1, self.nH_loc, 0, self.hd_loc), dtype=np.float32)
        feed = {}
        for i in range(self.L_loc):
            feed["past_k_%d" % i] = empty
            feed["past_v_%d" % i] = empty
        return feed

    def _split_past(self, out):
        # out = [hidden, present_k_0..L-1, present_v_0..L-1]
        return out[1:1 + self.L_loc], out[1 + self.L_loc:1 + 2 * self.L_loc]

    def _past_feed(self, pk, pv):
        feed = {}
        for i in range(self.L_loc):
            feed["past_k_%d" % i] = pk[i]
            feed["past_v_%d" % i] = pv[i]
        return feed

    def _acoustic_frame(self, h, temperature, top_k, top_p, rep_pen, hist):
        """h: (1, H) backbone hidden -> (codes, eos) for one 16-codebook frame."""
        H = self.hidden
        cond = h[0] if h.ndim == 2 else h[0, 0]
        cond = cond.astype(np.float32)
        txt = self.text_emb[self.sgs].astype(np.float32)
        tok = np.stack([cond, txt])[None].astype(np.float32)   # (1, 2, H)
        feed = {"token_emb": tok, "position_ids": np.array([[0, 1]], dtype=np.int64)}
        feed.update(self._empty_past())
        out = self.sess_ac.run(None, feed)
        hidden = out[0]
        pk, pv = self._split_past(out)
        slot0 = hidden[0, 0]

        def samp(ch, vec):
            logits = vec.astype(np.float32) @ self.audio_emb[ch].T   # (Va,)
            prev = hist[ch] if hist is not None else None
            code = self._sample(logits, temperature, top_k, top_p, rep_pen, prev)
            if hist is not None:
                hist[ch].add(code)
            return code

        codes = [samp(0, hidden[0, 1])]
        for ch in range(1, self.n_vq):
            emb = self.audio_emb[ch - 1][codes[-1]].astype(np.float32)
            feed = {"token_emb": emb.reshape(1, 1, H),
                    "position_ids": np.array([[ch + 1]], dtype=np.int64)}
            feed.update(self._past_feed(pk, pv))
            out = self.sess_ac.run(None, feed)
            hidden = out[0]
            pk, pv = self._split_past(out)
            codes.append(samp(ch, hidden[0, 0]))
        text_logits = slot0.astype(np.float32) @ self.text_emb.T
        eos = int(text_logits.argmax()) == self.eos_speech
        return codes, eos

    # -- codec (MOSS ONNX) -------------------------------------------------------
    def _decode_codes(self, codes: np.ndarray) -> np.ndarray:
        c = np.asarray(codes, dtype=np.int32)[None]      # (1, T, n_vq); codec wants int32
        lens = np.array([c.shape[1]], dtype=np.int32)
        out = self.sess_codec_dec.run(None, {"audio_codes": c, "audio_code_lengths": lens})
        return out[0][0].mean(0).astype(np.float32)

    # -- non-streaming synthesis -------------------------------------------------
    def synth(self, phonemes: str, temperature: float = 0.8, top_k: int = 25,
              top_p: float = 0.95, max_new_frames: int = 300,
              repetition_penalty: float = 1.2,
              repetition_window: int = DEFAULT_REP_WINDOW) -> np.ndarray:
        """Phoneme text -> full 48 kHz waveform (float32, may be empty)."""
        frames = list(self.stream_frames(
            phonemes, temperature, top_k, top_p, max_new_frames,
            repetition_penalty, repetition_window))
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return self._decode_codes(np.stack(frames))

    # -- shared frame-generation loop (generator over (T, n_vq) frames) ----------
    def stream_frames(self, phonemes: str, temperature: float = 0.8,
                      top_k: int = 25, top_p: float = 0.95,
                      max_new_frames: int = 300, repetition_penalty: float = 1.2,
                      repetition_window: int = DEFAULT_REP_WINDOW
                      ) -> Generator[np.ndarray, None, None]:
        """Yields (n_vq,) int frames as they are generated (frame level)."""
        if self.voice is None:
            raise RuntimeError("no voice selected (set_voice / default voice missing)")
        speaker_emb, ref_codes = self.voice
        anchor = self._speaker_anchor(speaker_emb)
        rows = self._build_rows(phonemes, ref_codes)
        prompt_embeds = self._embed_rows(rows, anchor)

        with self._lock:
            pre = self.sess_pre.run(None, {"inputs_embeds": prompt_embeds})
            past_k = [pre[1 + i] for i in range(self.L)]
            past_v = [pre[1 + self.L + i] for i in range(self.L)]
            h = pre[0][:, -1]
            Tprompt = prompt_embeds.shape[1]
            hist = (RepetitionHistory(self.n_vq, repetition_window)
                    if not math.isclose(repetition_penalty, 1.0) else None)

            for t in range(max_new_frames):
                codes, eos = self._acoustic_frame(h, temperature, top_k, top_p,
                                                  repetition_penalty, hist)
                yield np.asarray(codes, dtype=np.int64)
                if eos:
                    break
                slot = np.full((1, 1, self.n_vq + 1), self.audio_pad, dtype=np.int64)
                slot[:, :, 0] = self.sgs
                slot[0, 0, 1:] = codes
                se = self._embed_rows(slot[0], anchor)
                feed = {"inputs_embeds": se,
                        "position_ids": np.array([[Tprompt + t]], dtype=np.int64)}
                for i in range(self.L):
                    feed["past_k_%d" % i] = past_k[i]
                    feed["past_v_%d" % i] = past_v[i]
                out = self.sess_dec.run(None, feed)
                h = out[0][:, 0]
                past_k = [out[1 + i] for i in range(self.L)]
                past_v = [out[1 + self.L + i] for i in range(self.L)]

    # -- streaming synthesis (speech-token -> wav incrementally) ------------------
    def _stream_new_state(self) -> dict:
        """Fresh decode_step state (cached_positions init = -1 marks empty slots)."""
        sd_spec = self._codec_stream_spec
        st: dict = {}
        for t in sd_spec["transformer_offsets"]:
            st[t["input_name"]] = np.zeros(tuple(t["shape"]), np.int32)
        for a in sd_spec["attention_caches"]:
            st[a["offset_input_name"]] = np.zeros(tuple(a["offset_shape"]), np.int32)
            st[a["cached_keys_input_name"]] = np.zeros(tuple(a["cache_shape"]), np.float32)
            st[a["cached_values_input_name"]] = np.zeros(tuple(a["cache_shape"]), np.float32)
            st[a["cached_positions_input_name"]] = np.full(tuple(a["positions_shape"]), -1, np.int32)
        return st

    def _stream_decode(self, frames: np.ndarray, state: dict) -> np.ndarray:
        """Decode a group of frames (K, n_vq) incrementally, state in place."""
        sd_spec = self._codec_stream_spec
        codes = np.asarray(frames, dtype=np.int32)[None]          # (1, K, n_vq)
        feed = {"audio_codes": codes, "audio_code_lengths": np.array([codes.shape[1]], np.int32)}
        feed.update(state)
        outs = self.sess_codec_step.run(None, feed)
        d = dict(zip(self._codec_step_out_names, outs))
        for t in sd_spec["transformer_offsets"]:
            state[t["input_name"]] = d[t["output_name"]]
        for a in sd_spec["attention_caches"]:
            state[a["offset_input_name"]] = d[a["offset_output_name"]]
            state[a["cached_keys_input_name"]] = d[a["cached_keys_output_name"]]
            state[a["cached_values_input_name"]] = d[a["cached_values_output_name"]]
            state[a["cached_positions_input_name"]] = d[a["cached_positions_output_name"]]
        return d["audio"][0].mean(0)[: int(d["audio_lengths"][0])].astype(np.float32)

    def synth_stream(self, phonemes: str, chunk_frames: int = 25,
                     temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
                     max_new_frames: int = 300, repetition_penalty: float = 1.2,
                     repetition_window: int = DEFAULT_REP_WINDOW
                     ) -> Generator[np.ndarray, None, None]:
        """Phoneme text -> 48 kHz audio chunks as frames are decoded."""
        if self.sess_codec_step is None or self._codec_stream_spec is None:
            yield self.synth(phonemes, temperature, top_k, top_p,
                             max_new_frames, repetition_penalty, repetition_window)
            return
        state = self._stream_new_state()
        buffer: List[np.ndarray] = []
        for frame in self.stream_frames(phonemes, temperature, top_k, top_p,
                                        max_new_frames, repetition_penalty,
                                        repetition_window):
            buffer.append(frame)
            if len(buffer) >= max(1, chunk_frames):
                yield self._stream_decode(np.stack(buffer), state)
                buffer = []
        if buffer:
            yield self._stream_decode(np.stack(buffer), state)


def os_cpu() -> Optional[int]:
    import os
    return os.cpu_count()
