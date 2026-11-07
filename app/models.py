"""Local embedding and reranking models (sentence-transformers, CPU by default).

The Docker image downloads both models at build time and runs with
``HF_HUB_OFFLINE=1``, so the API makes no calls to the Hugging Face Hub.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from .config import get_settings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model %s", model_name)
        self.name = model_name
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self._model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        )
        return vector[0].astype(np.float32)


class Reranker:
    """Cross-encoder: reads the query and a passage together and scores relevance."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        log.info("loading reranker model %s", model_name)
        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages], batch_size=batch_size, convert_to_numpy=True
        )
        return [float(s) for s in np.asarray(scores).reshape(-1)]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return Embedder(s.embed_model, s.embed_query_prefix)


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings().rerank_model)
