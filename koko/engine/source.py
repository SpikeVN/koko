"""Microphone capture stage (own thread).

Pulls small blocks from PortAudio, packs blocks into ASR-sized chunks,
checks an RMS energy threshold for speech, and publishes:
  SOURCE_CHUNK  -> ASR stage  (numpy waveform on `payload`)
  SOURCE_SPEAKS -> gate       (`payload` = {"active": bool, "dur_s": seconds})

Publishes via bus.put(), which is safe from any thread; the bus pump
fans events out to each stage's asyncio queue.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from koko.engine.bus import Bus
from koko.engine.config import Config
from koko.engine.events import Event, Kind
from koko.engine.monitor import Monitor

log = logging.getLogger("koko.source")

DBFS_FLOOR = -45.0  # below this RMS the block counts as silence


def _dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x))))
    return 20.0 * np.log10(rms + 1e-9)


class MicSource:
    def __init__(self, cfg: Config, bus: Bus, monitor: Monitor):
        self.cfg = cfg.source
        self.bus = bus
        self.monitor = monitor

    def run_forever(self, stop: threading.Event) -> None:
        block_asr = int(self.cfg.whisper_chunk_s * self.cfg.sample_rate)
        block_sub = int(0.02 * self.cfg.sample_rate)  # 20 ms sub-blocks
        stream = sd.InputStream(
            channels=1,
            blocksize=block_sub,
            samplerate=self.cfg.sample_rate,
            device=self.cfg.device,
        )
        packed: list[np.ndarray] = []
        log.info(
            "mic streaming @%d Hz, chunk=%.1f s (%d-sample sub-blocks)",
            self.cfg.sample_rate, self.cfg.whisper_chunk_s, block_sub,
        )
        try:
            with stream:
                while not stop.is_set():
                    data, _ = stream.read(block_sub)
                    mono = data[:, 0].astype(np.float32, copy=True)
                    packed.append(mono)
                    if len(packed) * block_sub >= block_asr:
                        chunk = np.concatenate(packed)
                        packed = []
                        self.bus.put(Event(kind=Kind.SOURCE_CHUNK, payload=chunk))
                        db = _dbfs(chunk)
                        self.bus.put(Event(
                            kind=Kind.SOURCE_SPEAKS,
                            payload={"active": bool(db > DBFS_FLOOR),
                                     "dur_s": chunk.size / self.cfg.sample_rate},
                        ))
        except Exception:
            log.exception("source crashed")
