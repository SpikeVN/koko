#!/usr/bin/env python3
"""Post-setup self-check: ASR import + model, phonemize endpoint, TTS load+synth.

Run from the repo root after tools/setup_laptop.sh:
  .venv-laptop/bin/python tools/check_laptop.py
Exit code 0 = everything green; details printed per stage.
"""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

ok = True


def check(name, fn):
    global ok
    t0 = time.perf_counter()
    try:
        detail = fn()
        print(f"OK   {name}: {detail} ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        ok = False
        print(f"FAIL {name}: {type(e).__name__}: {e}")


def asr():
    from engine.asr import WhisperLiveAsr
    from engine.config import load_config
    import asyncio
    import numpy as np

    cfg = load_config()
    transcriber = WhisperLiveAsr(cfg)

    async def _check():
        # One silent block drives the real path: connect -> handshake ->
        # expect SERVER_READY -> send -> drain. A failed handshake raises
        # loudly inside transcribe(), which surfaces as a FAIL below.
        silence = np.zeros(
            int(cfg.source.sample_rate * cfg.source.whisper_chunk_s),
            dtype=np.float32)
        return await transcriber.transcribe(silence)

    _, lang = asyncio.run(_check())
    a = cfg.asr
    return (f"handshake OK ({a.whisper_live_url}, model={a.whisper_live_model}, "
            f"lang={lang})")


def phonemize():
    import httpx
    r = httpx.post(
        "http://127.0.0.1:8788/phonemize", json={"text": "xin chào"}, timeout=10)
    r.raise_for_status()
    ph = r.json().get("phonemes")
    assert ph, "empty phonemes"
    return ph


def tts():
    from engine.tts_vieneu import VieneuLite
    eng = VieneuLite("tts_model")
    v = eng.default_voice or next(iter(eng.presets), "")
    eng.set_voice(v)
    # smoke synth: fixed phoneme string, greedy for determinism
    ph = "zˈaːɜ kˈo4 fˈiɛɜw t̪ˈaŋ."
    t0 = time.perf_counter()
    wav = eng.synth(ph, temperature=0.0)
    dur = len(wav) / eng.SAMPLE_RATE
    peak = float(abs(wav).max()) if wav.size else 0.0
    return f"voice={v}, {dur:.2f}s wav in {time.perf_counter() - t0:.2f}s, peak={peak:.2f}"


def tts_stream():
    from engine.tts_vieneu import VieneuLite
    eng = VieneuLite("tts_model")
    v = eng.default_voice or next(iter(eng.presets), "")
    eng.set_voice(v)
    chunks = list(eng.synth_stream(
        "zˈaːɜ kˈo4 fˈiɛɜw t̪ˈaŋ.", chunk_frames=25, temperature=0.0))
    return f"{sum(len(c) for c in chunks) / eng.SAMPLE_RATE:.2f}s in {len(chunks)} chunks"


check("whisper-live :9090", asr)
check("phonemize :8788", phonemize)
check("vieneu synth", tts)
check("vieneu synth_stream", tts_stream)

print()
print("ALL OK — models ready; see config.toml and run main.py"
      if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
