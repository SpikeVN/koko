#!/usr/bin/env python3
"""Provision the isolated WhisperLive environment and start its server."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import argparse

from koko.engine.config import load_config


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv-whisperlive"
REQUIREMENTS = ROOT / "deploy" / "jetson" / "requirements.whisper-live.txt"


def python_executable() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def provision(python_version: str) -> Path:
    python = python_executable()
    if not python.is_file():
        subprocess.run(
            ["uv", "venv", "--python", python_version, str(VENV)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "pip"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--ignore-requires-python",
             "--no-deps", "whisper-live==0.10.0", "faster-whisper==1.2.0"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python),
             "-r", str(REQUIREMENTS)],
            cwd=ROOT,
            check=True,
        )
        if python_version != "3.8" and os.name != "nt":
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python),
                 "--index-url", "https://download.pytorch.org/whl/cu121",
                 "torch==2.3.1", "torchaudio==2.3.1"],
                cwd=ROOT,
                check=True,
            )
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml",
                        help="path to the koko config file")
    parser.add_argument(
        "--setup", action="store_true",
        help="create and install the WhisperLive virtual environment, then exit",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    python = provision(cfg.asr.whisper_live_python)
    if args.setup:
        return 0
    site_packages = next((VENV / "lib").glob("python*/site-packages"))
    cuda_libs = [
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in cuda_libs), os.environ.get("LD_LIBRARY_PATH", "")]
    )
    return subprocess.run(
        [str(python), "-m", "koko.whisper_live_server", "--config", args.config],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
