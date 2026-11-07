"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    """Merge several ranked lists into one.

    Each item scores ``sum(1 / (k + rank))`` over the lists it appears in (ranks
    start at 1). Only positions matter, so scores from different retrievers
    (cosine similarity, ``ts_rank``) never need to be put on the same scale.
    Ties are broken by the best single rank, then by first appearance.
    """
    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    order: dict[Hashable, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            order.setdefault(item, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], order[kv[0]]))
