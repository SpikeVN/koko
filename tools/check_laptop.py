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
    from faster_whisper import WhisperModel
    import faster_whisper
    return faster_whisper.__version__ + " (model auto-downloads on first use)"


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


check("faster-whisper", asr)
check("phonemize :8788", phonemize)
check("vieneu synth", tts)
check("vieneu synth_stream", tts_stream)

print()
print("ALL OK — run main.py with KOKO_TTS=vieneu" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
