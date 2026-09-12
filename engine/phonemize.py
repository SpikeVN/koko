"""Remote phonemizer client.

sea-g2p's G2P core is a Rust abi3 extension (py3.10+ only, no cp38/aarch64
wheel), so text normalization + G2P runs on the cloud host (same box as the
LLM, reached over the tunnel). The TTS engine here receives phonemes only.

The endpoint must accept POST {"text": "..."} and return JSON
{"phonemes": "..."} — a ~15 line sea-g2p wrapper on the LLM box.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from engine.config import Config

log = logging.getLogger("koko.phonemize")


class Phonemizer:
    def __init__(self, cfg: Config, timeout: float = 10.0):
        self.url = cfg.phonemize_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def run(self, text: str) -> Optional[str]:
        """text -> phoneme string, or None (caller should still log/use raw)."""
        try:
            resp = await self._client.post(self.url, json={"text": text})
            resp.raise_for_status()
            data = resp.json()
            return data.get("phonemes")
        except Exception:
            log.exception("phonemize request failed (%s)", self.url)
            return None

    async def close(self) -> None:
        await self._client.aclose()
