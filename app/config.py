"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str
    api_key: str

    embed_model: str
    embed_query_prefix: str
    rerank_model: str

    llm_provider: str
    ollama_url: str
    ollama_model: str
    vllm_url: str
    vllm_model: str
    llm_timeout_s: int
    llm_max_tokens: int

    chunk_words: int
    chunk_overlap_words: int
    top_k: int
    candidates: int
    rerank_candidates: int
    rrf_k: int
    max_upload_mb: int

    @property
    def llm_base_url(self) -> str:
        base = self.vllm_url if self.llm_provider == "vllm" else self.ollama_url
        return base.rstrip("/") + "/v1"

    @property
    def llm_model(self) -> str:
        return self.vllm_model if self.llm_provider == "vllm" else self.ollama_model


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        database_url=_str("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"),
        api_key=_str("API_KEY", ""),
        embed_model=_str("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        # bge v1.5 models retrieve better when short queries carry this instruction.
        embed_query_prefix=_str(
            "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        ),
        rerank_model=_str("RERANK_MODEL", "BAAI/bge-reranker-base"),
        llm_provider=_str("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_url=_str("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=_str("OLLAMA_MODEL", "qwen3:4b-instruct"),
        vllm_url=_str("VLLM_URL", "http://localhost:8001"),
        vllm_model=_str("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
        llm_timeout_s=_int("LLM_TIMEOUT_S", 180),
        llm_max_tokens=_int("LLM_MAX_TOKENS", 400),
        chunk_words=_int("CHUNK_WORDS", 200),
        chunk_overlap_words=_int("CHUNK_OVERLAP_WORDS", 40),
        top_k=_int("TOP_K", 5),
        candidates=_int("CANDIDATES", 30),
        rerank_candidates=_int("RERANK_CANDIDATES", 20),
        rrf_k=_int("RRF_K", 60),
        max_upload_mb=_int("MAX_UPLOAD_MB", 25),
    )
    if settings.llm_provider not in ("ollama", "vllm"):
        raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'vllm', got {settings.llm_provider!r}")
    return settings
