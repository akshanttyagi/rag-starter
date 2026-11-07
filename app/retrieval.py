"""Retrieval: vector search, full-text search, rank fusion and reranking.

Modes, from simplest to best:

* ``vector``:  nearest neighbours by cosine distance (pgvector, HNSW index)
* ``keyword``: PostgreSQL full-text search ranked with ``ts_rank``
* ``hybrid``:  both lists merged with reciprocal rank fusion
* ``rerank``:  the hybrid shortlist re-scored by a cross-encoder (default)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from . import db
from .config import get_settings
from .fusion import reciprocal_rank_fusion
from .models import get_embedder, get_reranker

Mode = Literal["vector", "keyword", "hybrid", "rerank"]
MODES: tuple[str, ...] = ("vector", "keyword", "hybrid", "rerank")


@dataclass
class Hit:
    chunk_id: int
    source: str
    title: str
    page: int | None
    chunk_index: int
    content: str
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Natural-language questions rarely match every term, so the parsed query's
# AND operators are swapped for OR and ts_rank rewards chunks that match more terms.
_KEYWORD_SQL = """
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery AS query
)
SELECT c.id, ts_rank(c.tsv, q.query, 1) AS score
FROM chunks c, q
WHERE c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(limit)s
"""

_VECTOR_SQL = """
SELECT id, 1 - (embedding <=> %(vec)s) AS score
FROM chunks
ORDER BY embedding <=> %(vec)s, id
LIMIT %(limit)s
"""

_FETCH_SQL = """
SELECT c.id, d.source, d.title, c.page, c.chunk_index, c.content
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%(ids)s)
"""


def vector_search(query: str, limit: int) -> list[tuple[int, float]]:
    vec = get_embedder().embed_query(query)
    with db.pool().connection() as conn:
        # Exact search would also be fine at this size; ef_search keeps HNSW recall high.
        conn.execute("SET hnsw.ef_search = 100")
        rows = conn.execute(_VECTOR_SQL, {"vec": vec, "limit": limit}).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def keyword_search(query: str, limit: int) -> list[tuple[int, float]]:
    with db.pool().connection() as conn:
        rows = conn.execute(_KEYWORD_SQL, {"query": query, "limit": limit}).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def _fetch(ids: list[int]) -> dict[int, tuple]:
    if not ids:
        return {}
    with db.pool().connection() as conn:
        rows = conn.execute(_FETCH_SQL, {"ids": ids}).fetchall()
    return {int(r[0]): r for r in rows}


def _hits(ranked: list[tuple[int, float]], vec_rank: dict, kw_rank: dict) -> list[Hit]:
    rows = _fetch([cid for cid, _ in ranked])
    hits = []
    for cid, score in ranked:
        r = rows.get(cid)
        if r is None:  # deleted between search and fetch
            continue
        hits.append(
            Hit(
                chunk_id=cid,
                source=r[1],
                title=r[2],
                page=r[3],
                chunk_index=r[4],
                content=r[5],
                score=round(score, 6),
                vector_rank=vec_rank.get(cid),
                keyword_rank=kw_rank.get(cid),
            )
        )
    return hits


def search(query: str, mode: Mode = "rerank", top_k: int | None = None) -> list[Hit]:
    s = get_settings()
    top_k = top_k or s.top_k
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    vector = vector_search(query, s.candidates) if mode != "keyword" else []
    keyword = keyword_search(query, s.candidates) if mode != "vector" else []
    vec_rank = {cid: i for i, (cid, _) in enumerate(vector, start=1)}
    kw_rank = {cid: i for i, (cid, _) in enumerate(keyword, start=1)}

    if mode == "vector":
        return _hits(vector[:top_k], vec_rank, kw_rank)
    if mode == "keyword":
        return _hits(keyword[:top_k], vec_rank, kw_rank)

    fused = reciprocal_rank_fusion([[c for c, _ in vector], [c for c, _ in keyword]], k=s.rrf_k)
    if mode == "hybrid":
        return _hits([(c, sc) for c, sc in fused[:top_k]], vec_rank, kw_rank)

    shortlist = _hits(
        [(c, sc) for c, sc in fused[: max(s.rerank_candidates, top_k)]], vec_rank, kw_rank
    )
    scores = get_reranker().score(query, [h.content for h in shortlist])
    for hit, score in zip(shortlist, scores, strict=True):
        hit.score = round(score, 6)
    shortlist.sort(key=lambda h: h.score, reverse=True)
    return shortlist[:top_k]
