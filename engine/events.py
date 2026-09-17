"""Typed events crossing stage boundaries on the event bus."""

from __future__ import annotations

import time
import dataclasses
from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    SOURCE_CHUNK = "source_chunk"    # raw audio block from the mic
    SOURCE_SPEAKS = "source_speaks"  # speaker activity drives the gate
    PARTIAL_TEXT = "partial_text"    # provisional ASR output; display only
    FINAL_TEXT = "final_text"        # confirmed ASR output
    SPEAK = "speak"                  # gate -> LLM: interpret this
    ASSISTANT_CHUNK = "assistant_chunk"  # LLM sentence ready for TTS
    BARGE_IN = "barge_in"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    kind: Kind
    ts: float = field(default_factory=time.monotonic)
    turn_id: str = ""
    text: str = ""
    speaker: str = ""
    is_final: bool = False
    payload: object = None

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
