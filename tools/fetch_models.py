#!/usr/bin/env python3
"""Download the TTS model files from Hugging Face.

The Hugging Face ``hf`` command is used instead of shell utilities.  This
keeps the download layout identical on Windows, macOS, and Linux.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def cache_dir() -> Path:
    configured = os.environ.get("HF_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    external = Path("/mnt/sdcard/PhuongBase/downloads/hf_cache")
    if external.is_dir():
        return external
    return ROOT / "hf_cache"


def download(repo: str, includes: list[str], destination: Path, force: bool,
             strip_prefix: str = "") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="koko-hf-") as temp:
        executable = ["hf"] if shutil.which("hf") else [
            "uv", "run", "--with", "huggingface_hub", "hf"]
        command = executable + ["download", repo]
        for pattern in includes:
            command.extend(["--include", pattern])
        if force:
            command.append("--force-download")
        command.extend(["--local-dir", temp])
        env = os.environ.copy()
        env["HF_HOME"] = str(cache_dir())
        env["HF_HUB_CACHE"] = str(cache_dir())
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        source = Path(temp)
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            relative = item.relative_to(source)
            if relative.parts and relative.parts[0] == ".cache":
                continue
            if strip_prefix:
                prefix = Path(strip_prefix)
                if relative.parts[:len(prefix.parts)] != prefix.parts:
                    continue
                relative = relative.relative_to(prefix)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not force:
                    continue
                target.unlink()
            shutil.move(str(item), str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload existing files")
    args = parser.parse_args()
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"HF cache: {cache}")
    print("== Vieneu v3 Turbo: fp32 graphs ==")
    download(
        "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        ["onnx_update/*"],
        ROOT / "tts_model" / "onnx_int8",
        args.force,
        strip_prefix="onnx_update",
    )
    print("== Vieneu: speaker encoder + denoiser ==")
    download(
        "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        ["speaker_encoder.onnx", "denoiser.onnx"],
        ROOT / "tts_model",
        args.force,
    )
    print("== MOSS audio tokenizer ==")
    download(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
        ["moss_audio_tokenizer_*", "codec_browser_onnx_meta.json"],
        ROOT / "tts_model" / "codec",
        args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
