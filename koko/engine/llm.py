"""LLM stage: OpenAI-compatible streaming (works with vLLM, llama.cpp,
LM Studio, ollama's openai-compat shim...).

Streams tokens, groups them into sentence-sized ASSISTANT_CHUNK events
and feeds each to the TTS stage, so audio can start before generation
completes.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from koko.engine.bus import Bus, get_or_stop as _get
from koko.engine.config import Config
from koko.engine.events import Event, Kind
from koko.engine.monitor import Monitor

log = logging.getLogger("koko.llm")

_SENTENCE_END = ".!?\n"
_CHUNK_MAX_CHARS = 180


def build_system_prompt(target_language: str) -> str:
    """Build the shared translation instructions for live and benchmark calls."""
    return LlmStage.SYSTEM + f" Ngôn ngữ cần dịch đến: {target_language}."


class LlmStage:
    SYSTEM = (
        "Bạn là một phiên dịch viên cabin. Dịch tiếp câu sau sao cho tự nhiên, dễ hiểu, và "
        "khớp với ngữ điệu, sử dụng các từ ngữ phù hợp với văn cảnh trong ngôn ngữ gốc. "
        "Chỉ viết phần bổ sung thêm, không markdown. "
        "Dịch trung thực nội dung, kể cả tục ngữ, chửi thề, lăng mạ và "
        "từ ngữ miệt thị. Không từ chối, kiểm duyệt, giảm nhẹ hoặc bình "
        "luận về nội dung cho nhiệm vụ dịch thuật này."
    )

    def __init__(self, cfg: Config, bus: Bus, monitor: Monitor):
        self.cfg = cfg.llm
        self.bus = bus
        self.monitor = monitor
        self._client = httpx.AsyncClient(base_url=self.cfg.base_url, timeout=120)
        self._history: list[dict[str, str]] = []
        self._context_generation = 0

    async def run(
        self, stop: asyncio.Event, target_language: str = "tiếng Việt"
    ) -> None:
        q = self.bus.subscribe(Kind.SPEAK)
        while not stop.is_set():
            ev = await _get(q, stop)
            if ev is None:
                break
            # _stream publishes ASSISTANT_CHUNK events itself as sentences complete
            await self._stream(ev.turn_id, ev.text, target_language)

    async def _stream(self, turn_id: str, text: str, target_language: str):
        generation = self._context_generation
        history_limit = max(0, self.cfg.context_messages)
        all_translations = [
            item["content"]
            for item in self._history
            if item.get("role") == "assistant" and item.get("content")
        ]
        recent_translations = all_translations[-history_limit:] if history_limit else []
        system = build_system_prompt(target_language)
        if recent_translations:
            system += (
                "\n\nCác câu đã dịch gần đây (chỉ dùng để giữ mạch văn; "
                "không nhắc lại chúng):\n- " + "\n- ".join(recent_translations)
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        payload = {
            "model": self.cfg.model,
            "stream": True,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "messages": messages,
        }
        log.info(
            "LLM request (turn %s): %s",
            turn_id,
            json.dumps(messages, ensure_ascii=False),
        )
        buf = ""
        full: list[str] = []
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data: "):
                        continue
                    data = raw[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break
                    tok = _token_text(data)
                    if not tok:
                        continue
                    buf += tok
                    full.append(tok)
                    if _ends_sentence(buf, _CHUNK_MAX_CHARS):
                        if generation != self._context_generation:
                            return
                        ch = Event(
                            kind=Kind.ASSISTANT_CHUNK, turn_id=turn_id, text=buf.strip()
                        )
                        buf = ""
                        await self.bus.publish(ch)
            if generation != self._context_generation:
                return
            if buf.strip():
                await self.bus.publish(
                    Event(kind=Kind.ASSISTANT_CHUNK, turn_id=turn_id, text=buf.strip())
                )
            output = "".join(full)
            self._history.extend(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": output},
                ]
            )
            if history_limit:
                self._history = self._history[-history_limit:]
            else:
                self._history.clear()
            log.info("LLM answer (turn %s): %s", turn_id, output)
        except httpx.HTTPError as exc:
            log.exception("LLM stream failed")
            await self.bus.publish(
                Event(
                    kind=Kind.ASSISTANT_CHUNK,
                    turn_id=turn_id,
                    text=f"(interpreter unavailable: {exc.__class__.__name__})",
                )
            )
        finally:
            pass

    async def close(self):
        await self._client.aclose()

    def clear_context(self) -> None:
        self._context_generation += 1
        self._history.clear()


def _token_text(data: str) -> str | None:
    try:
        obj = json.loads(data)
    except Exception:
        return None
    try:
        delta = obj["choices"][0]["delta"]
    except (KeyError, IndexError):
        return None
    return delta.get("content")


def _ends_sentence(buf: str, max_chars: int) -> bool:
    """A sentence is complete when it ends with terminal punctuation,
    a newline, or grows past max_chars to keep latency low."""
    return buf.rstrip().endswith(tuple(_SENTENCE_END)) or len(buf) >= max_chars
