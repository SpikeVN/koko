#!/usr/bin/env python3
"""Provision a laptop checkout and run its local self-check.

This intentionally uses only Python and the ``uv`` executable.  In
particular, it does not assume POSIX venv paths, a shell, curl, or Unix
signals, so the same command works from PowerShell and from a Unix shell.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def run(command: list[str], **kwargs) -> None:
    print("==>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def venv_python() -> Path:
    name = "python.exe" if os.name == "nt" else "python"
    return VENV / ("Scripts" if os.name == "nt" else "bin") / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="Python interpreter/version passed to uv venv")
    args = parser.parse_args()

    command = ["uv", "venv", str(VENV)]
    if args.python:
        command.extend(["--python", args.python])
    run(command)
    run(["uv", "pip", "install", "--python", str(venv_python()), "-e", "."])
    print("== provisioning WhisperLive virtual environment ==")
    run([str(venv_python()), str(ROOT / "whisper_live.py"), "--setup"])
    run([str(venv_python()), str(ROOT / "tools" / "fetch_models.py")])
    result = subprocess.run(
        [str(venv_python()), str(ROOT / "tools" / "check_laptop.py")],
        cwd=ROOT,
    )
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode or 1)
