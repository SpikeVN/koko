"""CLI: run the interpreter pipeline (local, non-websocket).

  uv run python main.py                  # default: release_words gate

All knobs live in config.toml (loaded by engine/config.load_config); there
are no environment variables. Edit config.toml or pass a per-machine file.
"""

import logging

from engine.pipeline import main


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
