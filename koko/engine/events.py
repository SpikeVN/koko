"""Typed events crossing stage boundaries on the event bus."""

from __future__ import annotations

import time
import dataclasses
from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    SOURCE_CHUNK = "source_chunk"
    SOURCE_SPEAKS = "source_speaks"
    PARTIAL_TEXT = "partial_text"
    FINAL_TEXT = "final_text"
    SPEAK = "speak"
    ASSISTANT_PARTIAL = "assistant_partial"
    ASSISTANT_CHUNK = "assistant_chunk"
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
