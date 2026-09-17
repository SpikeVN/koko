#!/usr/bin/env python3
"""Provision the isolated WhisperLive environment and start its server."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import argparse


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv-whisperlive"


def python_executable() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def provision() -> Path:
    python = python_executable()
    if not python.is_file():
        subprocess.run(["uv", "venv", str(VENV)], cwd=ROOT, check=True)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-build-isolation",
             "setuptools<81", "whisper-live"],
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
    python = provision()
    if args.setup:
        return 0
    return subprocess.run(
        [str(python), "-m", "koko.whisper_live_server", "--config", args.config],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
