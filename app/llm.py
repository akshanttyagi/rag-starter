"""Answer generation through an OpenAI-compatible chat endpoint.

Ollama (``/v1``) and vLLM both speak the same chat-completions API, so the only
difference between the CPU and GPU setups is the base URL and the model name.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from .config import Settings, get_settings

log = logging.getLogger(__name__)

NOT_FOUND = "I could not find this in the documents."

SYSTEM_PROMPT = (
    "You are a document assistant. Answer only from the numbered sources in the user's "
    "message, never from your own knowledge. After each fact, cite its source number in "
    "square brackets, like [1] or [2][3]. Keep answers to one to three sentences. If the "
    f"sources do not answer the question, reply exactly: {NOT_FOUND}"
)

# One worked example of each case (answer with a citation, and refuse) teaches small
# models the format far more reliably than instructions alone.
_EXAMPLE_SOURCES = """[1] (handbook.pdf, page 4)
Employees receive 20 days of paid leave per year. Unused days can be carried over until March 31.

[2] (handbook.pdf, page 9)
Remote work requires written approval from a manager."""

_EXAMPLES = [
    ("How many days of paid leave do employees get?",
     "Employees get 20 days of paid leave per year, and unused days carry over until March 31 [1]."),
    ("What is the dress code?", NOT_FOUND),
]  # fmt: skip


def _user_turn(sources: str, question: str) -> str:
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def build_messages(question: str, passages: list[tuple[str, str]]) -> list[dict]:
    """``passages`` are (label, text) pairs, numbered from 1 in the prompt."""
    blocks = "\n\n".join(f"[{i}] ({label})\n{text}" for i, (label, text) in enumerate(passages, 1))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_question, example_answer in _EXAMPLES:
        messages.append({"role": "user", "content": _user_turn(_EXAMPLE_SOURCES, example_question)})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": _user_turn(blocks, question)})
    return messages


_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    """Source numbers referenced in the answer, in order of first use."""
    seen: list[int] = []
    for group in _CITATION.findall(answer):
        for part in re.split(r"[,;]", group):
            n = int(part)
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen


def chat(messages: list[dict], settings: Settings | None = None) -> str:
    s = settings or get_settings()
    response = httpx.post(
        f"{s.llm_base_url}/chat/completions",
        json={
            "model": s.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": s.llm_max_tokens,
        },
        timeout=s.llm_timeout_s,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


@dataclass
class LLMStatus:
    ready: bool = False
    detail: str = "starting"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, ready: bool, detail: str) -> None:
        with self._lock:
            self.ready, self.detail = ready, detail


status = LLMStatus()


def _ollama_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def ensure_model(settings: Settings | None = None) -> None:
    """Make sure the chat model can be served; pull it into Ollama if missing."""
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        base = s.ollama_url.rstrip("/")
        tags = httpx.get(f"{base}/api/tags", timeout=10).json().get("models", [])
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        if _ollama_tag(s.ollama_model) not in names:
            status.set(False, f"downloading {s.ollama_model} into Ollama")
            log.info("pulling %s into Ollama (first start only)", s.ollama_model)
            r = httpx.post(
                f"{base}/api/pull", json={"model": s.ollama_model, "stream": False}, timeout=None
            )
            r.raise_for_status()
    else:
        served = httpx.get(f"{s.llm_base_url}/models", timeout=10).json().get("data", [])
        ids = [m.get("id") for m in served]
        if s.llm_model not in ids:
            raise RuntimeError(f"vLLM serves {ids}, expected {s.llm_model}")
    status.set(True, "ready")


def watch_model(settings: Settings | None = None, retry_s: float = 5.0) -> None:
    """Retry ``ensure_model`` until it succeeds (vLLM can take minutes to load)."""
    while True:
        try:
            ensure_model(settings)
            log.info("LLM ready")
            return
        except Exception as exc:  # keep retrying; the reason shows in /health
            status.set(False, f"waiting for LLM server: {exc}")
            log.info("LLM not ready yet: %s", exc)
            time.sleep(retry_s)
