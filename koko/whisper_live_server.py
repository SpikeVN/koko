"""WhisperLive ASR server — config.toml-driven, zero CLI switches.

Parameters that used to be argparse flags (host, port, backend, model, max
clients, connection cap) now come from the `[asr]` section of config.toml
(koko/engine/config.py is the single source of truth). Run via
`python whisper_live.py`, which provisions the dedicated
`.venv-whisperlive` venv and boots this script.
"""
from __future__ import annotations

import logging
import os
import argparse
import sys
from pathlib import Path

from koko.engine.config import load_config

logging.basicConfig(level=logging.INFO)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml",
                        help="path to the koko config file")
    args = parser.parse_args()
    # Avoid ctranslate2 spinning up a thread per core by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cfg = load_config(args.config).asr

    # The repository bootstrap script is named ``whisper_live.py``.  The
    # service runs from the repository root, so that file would shadow the
    # third-party ``whisper_live`` package installed in the isolated venv.
    root = Path(__file__).resolve().parents[1]
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or ".").resolve() != root]
    from whisper_live.server import TranscriptionServer

    server = TranscriptionServer()
    server.run(
        cfg.whisper_live_host,
        port=cfg.whisper_live_port,
        backend=cfg.whisper_live_backend,
        faster_whisper_custom_model_path=cfg.whisper_live_model,
        default_model=cfg.whisper_live_model,   # REST fallback + health
        single_model=True,                       # load once, reuse per client
        max_clients=cfg.whisper_live_max_clients,
        max_connection_time=cfg.whisper_live_max_connection_s,
    )


if __name__ == "__main__":
    main()
