"""WhisperLive ASR server — config.toml-driven, zero CLI switches.

Parameters that used to be argparse flags (host, port, backend, model, max
clients, connection cap) now come from the `[asr]` section of config.toml
(engine/config.py is the single source of truth). Run via ./whisper-live.sh,
which provisions the dedicated .venv-whisperlive venv and boots this script.
"""
from __future__ import annotations

import logging
import os

from engine.config import load_config

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Avoid ctranslate2 spinning up a thread per core by default.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cfg = load_config().asr

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
