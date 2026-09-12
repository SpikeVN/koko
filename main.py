"""CLI: run the interpreter pipeline.

  uv run python main.py                  # default: 5s interpretation delay
  KOKO_DELAY_S=2 KOKO_TTS=vieneu uv run python main.py

All knobs are in engine/config.py or environment variables (KOKO_*).
"""

import logging

from engine.pipeline import main


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
