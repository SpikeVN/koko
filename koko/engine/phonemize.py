"""Remote phonemizer client."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from koko.engine.config import Config

log = logging.getLogger("koko.phonemize")


class Phonemizer:
    def __init__(self, cfg: Config, timeout: float = 10.0):
        self.url = cfg.phonemize_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def run(self, text: str) -> Optional[str]:
        """Return phonemes for text, or None when the service fails."""
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
