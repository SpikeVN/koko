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

import ctypes
import json
import logging
import math
import queue
import threading
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
from collections import Counter, deque

log = logging.getLogger("koko.tts_vieneu")

DEFAULT_REP_WINDOW = 64        # ~2.5 s audio at 25 frames/s: catches local loops only
_STREAM_LEADIN_FRAMES = 4

ORT = None  # lazy: imported in VieneuLite.__init__ so --help style imports stay fast


def _load_cuda_libs() -> bool:
    """Preload CUDA runtime libs RTLD_GLOBAL so ORT's provider dlopen resolves
    them even when LD_LIBRARY_PATH doesn't include the CUDA lib dir.
    Portable: absolute paths if /usr/local/cuda/lib64 exists (Jetson CUDA
    11.4), otherwise plain sonames for CUDA 12/13 (laptop). Returns True if
    the first candidate of each core family loads."""
    import os
    base = "/usr/local/cuda/lib64"
    candidates = ("libcudart.so.11.0", "libcublasLt.so.11", "libcublas.so.11",
                  "libcufft.so.10",
                  "libcudart.so.12.0", "libcublasLt.so.12.0", "libcublas.so.12.0",
                  "libcufft.so.11.0",
                  "libcudart.so.13.0", "libcublasLt.so.13.0", "libcublas.so.13.0",
                  "libcufft.so.12.0")
    found = {"cudart": False, "cublas": False, "cufft": False}
    for name in candidates:
        fam = ("cudart" if "cudart" in name
               else "cublas" if "cublasLt" in name
               else "cublas" if "cublas" in name
               else "cufft" if "cufft" in name else None)
        if fam in found and found[fam]:
            continue
        for path in (os.path.join(base, name), name):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            log.debug("cuda preload ok: %s", path)
            if fam in found:
                found[fam] = True
            break
    return all(found.values())


def _providers(mode: str = "auto") -> "tuple[str, ...]":
    """EP choice: auto|cuda|cpu (auto = CUDA if usable).

    `mode` comes from `tts.execution_provider` in config.toml."""
    mode = (mode or "auto").lower()
    if mode == "cpu":
        return ("CPUExecutionProvider",)
    cuda_ok = _load_cuda_libs()
    if mode == "cuda" and not cuda_ok:
        raise RuntimeError("execution_provider=cuda but CUDA libs failed to load")
    if cuda_ok:
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


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

    def __init__(self, model_dir: str, voice: str = "", threads: int = 0,
                 execution_provider: str = "auto", voices_path: str = ""):
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

        # sgs embedding + pre-transposed audio emb (used every acoustic frame)
        self.sgs_emb = self.text_emb[self.sgs]                 # (H,) float32
        self.text_emb_T = np.ascontiguousarray(self.text_emb.T)   # (H, Vt)
        self.audio_emb_T = np.ascontiguousarray(
            self.audio_emb.transpose(0, 2, 1))                 # (n_vq, H, Va)

        # -- phoneme tokenizer (tokenizers lib, torch-free) ------------------
        self.tokenizer = Tokenizer.from_file(str(vd / "tokenizer.json"))

        # -- ONNX sessions ----------------------------------------------------
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.inter_op_num_threads = 1
        if threads and threads > 0:
            intra = int(threads)
        else:
            intra = min(max((os_cpu() or 8) // 2, 1), 8)
        so.intra_op_num_threads = intra
        self.ort_intra_op_threads = intra
        prov = list(_providers(execution_provider))

        def _sess(path, p, o=None):
            log.debug("loading ONNX %s", path.name)
            return ort.InferenceSession(str(path), o if o is not None else so,
                                        providers=p)

        self.sess_pre = _sess(vd / "vieneu_prefill.onnx", prov)
        self.sess_dec = _sess(vd / "vieneu_decode_step.onnx", prov)
        # acoustic head is an int8 graph: CUDA session would just add DMA
        # overhead around its CPU fall-back nodes, so pin it to CPU. It gets
        # its own session options: spinning only for this tiny per-frame graph
        # (global spinning steals CPU from CUDA host-side work).
        so_ac = ort.SessionOptions()
        so_ac.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so_ac.inter_op_num_threads = 1
        so_ac.intra_op_num_threads = 4
        so_ac.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.sess_ac = _sess(vd / "vieneu_acoustic_cached.onnx",
                             ["CPUExecutionProvider"], so_ac)
        self._ac_cache_init()
        self.sess_codec_dec = _sess(cd / "moss_audio_tokenizer_decode_full.onnx",
                                    prov)

        # streaming codec decoder (decode_step) for streaming synth
        self.sess_codec_step = None
        self._codec_stream_spec = None
        try:
            meta = json.loads((cd / "codec_browser_onnx_meta.json").read_text(encoding="utf-8"))
            self._codec_stream_spec = meta.get("streaming_decode")
            self.sess_codec_step = _sess(cd / "moss_audio_tokenizer_decode_step.onnx",
                                         ["CPUExecutionProvider"])
            self._codec_step_out_names = [o.name for o in self.sess_codec_step.get_outputs()]
        except Exception:
            self.sess_codec_step = None
            log.debug("streaming codec unavailable, infer_stream will fall back")

        # -- preset voices ----------------------------------------------------
        voices_p = Path(voices_path).expanduser() if voices_path else (
            base / "voices_v3_turbo.json")
        self.voices_path = voices_p
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
        return emb[None].astype(np.float32, copy=False)

    def _sample(self, logits, temperature, top_k, top_p, rep_pen, prev):
        logits = logits.astype(np.float32, copy=False)
        if not math.isclose(rep_pen, 1.0) and prev:
            idx = np.fromiter(prev, dtype=np.int64, count=len(prev))
            sel = logits[idx]
            logits = logits.copy()
            logits[idx] = np.where(sel < 0, sel * rep_pen, sel / rep_pen)
        if not (temperature and temperature > 0):
            return int(logits.argmax())
        logits = logits / temperature
        V = logits.shape[-1]
        # top-k FIRST so the softmax below runs over k candidates only.
        if top_k and 0 < int(top_k) < V:
            k = int(top_k)
            cand = np.argpartition(logits, -k)[-k:]
        else:
            cand = np.arange(V)
        cs = logits[cand]
        order = np.argsort(cs)[::-1]
        cand = cand[order]
        x = cs[order]
        p = _softmax(x)
        if top_p and top_p < 1.0:
            keep = (np.cumsum(p) - p) < top_p
            p = p * keep
        cdf = p.cumsum()
        cdf /= cdf[-1]
        pos = min(int(cdf.searchsorted(np.random.random_sample(), side="right")),
                  p.shape[-1] - 1)
        return int(cand[pos])

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
    def _ac_cache_init(self):
        """Persistent buffers + IOBinding targets for the acoustic loop."""
        H = self.hidden
        nH, hd = self.nH_loc, self.hd_loc
        P = 2 + self.n_vq                       # max present length (<=17)
        self._ac_tok0 = np.zeros((1, 2, H), np.float32)   # step0: cond + sgs
        self._ac_tok0[0, 1] = self.sgs_emb
        self._ac_tok = np.zeros((1, 1, H), np.float32)    # steps 1..n_vq-1
        # sampling head scratch buffer: logits = vec @ audio_emb_T[ch]
        self._ac_logits = np.zeros(self.audio_emb.shape[1], np.float32)
        self._ac_text_logits = np.zeros(self.text_emb.shape[0], np.float32)
        self._ac_pos0 = np.array([[0, 1]], np.int64)
        self._ac_pos = np.zeros((1, 1), np.int64)
        self._ac_h0 = np.zeros((1, 2, H), np.float32)
        self._ac_h = np.zeros((1, 1, H), np.float32)
        # per layer: ((k ping, k pong), (v ping, v pong)); present is written
        # into one buffer while the previous step's buffer is read as past.
        self._ac_kv = [((np.zeros((1, nH, P, hd), np.float32),
                         np.zeros((1, nH, P, hd), np.float32)),
                        (np.zeros((1, nH, P, hd), np.float32),
                         np.zeros((1, nH, P, hd), np.float32)))
                       for _ in range(self.L_loc)]

    def _ac_bind_in(self, b, tok, pos, past_len, cur):
        b.bind_input("token_emb", "cpu", 0, np.float32, tok.shape,
                     tok.ctypes.data)
        b.bind_input("position_ids", "cpu", 0, np.int64, pos.shape,
                     pos.ctypes.data)
        for i in range(self.L_loc):
            kbuf, vbuf = self._ac_kv[i][0][cur], self._ac_kv[i][1][cur]
            k = kbuf[:, :, :past_len, :]
            b.bind_input("past_k_%d" % i, "cpu", 0, np.float32, k.shape,
                         k.ctypes.data)
            b.bind_input("past_v_%d" % i, "cpu", 0, np.float32, k.shape,
                         vbuf.ctypes.data)

    def _ac_bind_out(self, b, n_tok, past_len, cur):
        """Bind outputs; present lands in buffer[cur] (the *other* ping one)."""
        H = self.hidden
        nH, hd = self.nH_loc, self.hd_loc
        hid = self._ac_h0 if n_tok == 2 else self._ac_h
        b.bind_output("hidden", "cpu", 0, np.float32, (1, n_tok, H),
                      hid.ctypes.data)
        for i in range(self.L_loc):
            kbuf = self._ac_kv[i][0][cur]
            vbuf = self._ac_kv[i][1][cur]
            s = (1, nH, past_len + n_tok, hd)
            b.bind_output("present_k_%d" % i, "cpu", 0, np.float32, s,
                          kbuf.ctypes.data)
            b.bind_output("present_v_%d" % i, "cpu", 0, np.float32, s,
                          vbuf.ctypes.data)

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
        """h: (1, H) backbone hidden -> (codes, eos) for one 16-codebook frame.

        All runs go through IOBinding with persistent buffers (no per-run
        input copies / output allocations); KV ping-pongs between two
        pre-allocated buffers each growing by 1 slot per step.
        """
        sess = self.sess_ac
        b = sess.io_binding()
        cond = h[0] if h.ndim == 2 else h[0, 0]     # already float32
        self._ac_tok0[0, 0] = cond
        nH, hd = self.nH_loc, self.hd_loc
        cur = 0                                      # buffer holding "past"

        def samp(ch, vec):
            # single-threaded contraction: multi-threaded BLAS gemv thrashes
            # under CPU contention (~7 ms vs ~0.3 ms per call on this box)
            logits = np.einsum("i,ij->j", vec, self.audio_emb_T[ch],
                               out=self._ac_logits)
            prev = hist[ch] if hist is not None else None
            code = self._sample(logits, temperature, top_k, top_p, rep_pen, prev)
            if hist is not None:
                hist[ch].add(code)
            return code

        # -- step 0: 2 tokens, empty past ------------------------------------
        self._ac_bind_in(b, self._ac_tok0, self._ac_pos0, 0, cur)
        self._ac_bind_out(b, 2, 0, 1 - cur)
        sess.run_with_iobinding(b, None)
        slot0 = self._ac_h0[0, 0]
        codes = [samp(0, self._ac_h0[0, 1])]
        cur, past_len = 1 - cur, 2

        # -- steps 1..n_vq-1: 1 token, growing past ----------------------------
        for ch in range(1, self.n_vq):
            self._ac_tok[0, 0] = self.audio_emb[ch - 1][codes[-1]]
            self._ac_pos[0, 0] = ch + 1
            self._ac_bind_in(b, self._ac_tok, self._ac_pos, past_len, cur)
            self._ac_bind_out(b, 1, past_len, 1 - cur)
            sess.run_with_iobinding(b, None)
            past_len += 1
            codes.append(samp(ch, self._ac_h[0, 0]))
            cur = 1 - cur

        text_logits = np.einsum("i,ij->j", slot0, self.text_emb_T,
                                out=self._ac_text_logits)
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
                     repetition_window: int = DEFAULT_REP_WINDOW,
                     codec_thread: bool = True
                     ) -> Generator[np.ndarray, None, None]:
        """Phoneme text -> 48 kHz audio chunks as frames are decoded.

        Codec decode runs on its own worker thread (pure downstream sink —
        nothing the codec produces feeds back into generation), so it is
        fully overlapped with frame generation instead of running serially
        after each chunk-frame group. Falls back to inline decoding when
        codec_thread is False or the streaming codec is unavailable.
        """
        if self.sess_codec_step is None or self._codec_stream_spec is None:
            yield self.synth(phonemes, temperature, top_k, top_p,
                             max_new_frames, repetition_penalty, repetition_window)
            return
        if not codec_thread:
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
            return
        # codec consumer thread: frames -> mono float audio chunks (FIFO)
        frames_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=8)
        out_q: "queue.Queue[object]" = queue.Queue()

        def _codec_worker() -> None:
            try:
                state = self._stream_new_state()
                buffer: List[np.ndarray] = []
                while True:
                    f = frames_q.get()
                    if f is None:
                        break
                    buffer.append(f)
                    if len(buffer) >= max(1, chunk_frames):
                        out_q.put(self._stream_decode(np.stack(buffer), state))
                        buffer = []
                if buffer:
                    out_q.put(self._stream_decode(np.stack(buffer), state))
                out_q.put(None)  # done sentinel
            except Exception as e:  # propagate to consumer
                out_q.put(e)

        t = threading.Thread(target=_codec_worker, name="koko-codec-decode",
                             daemon=True)
        t.start()
        try:
            for frame in self.stream_frames(phonemes, temperature, top_k, top_p,
                                            max_new_frames, repetition_penalty,
                                            repetition_window):
                frames_q.put(frame)  # blocks if codec falls behind
            frames_q.put(None)
            while True:
                item = out_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            t.join(timeout=5.0)


def os_cpu() -> Optional[int]:
    import os
    return os.cpu_count()
