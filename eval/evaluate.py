"""Retrieval evaluation: vector-only vs. hybrid vs. hybrid + rerank.

Each question in ``questions.jsonl`` names the file that answers it and a short
phrase from the passage that does. A retrieved chunk counts as a hit when it
comes from that file and contains the phrase. The script calls the running
API's /search endpoint once per question and mode, and reports:

* hit@k: share of questions with a correct chunk in the top k
* MRR:   mean of 1 / rank of the first correct chunk (0 if none in the top k)

Run inside the API container (no extra setup):

    docker compose exec api python -m eval.evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
LABELS = {
    "vector": "vector only",
    "keyword": "full-text only",
    "hybrid": "hybrid (RRF)",
    "rerank": "hybrid + rerank",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u2019", "'")).strip().lower()


def first_hit_rank(results: list[dict], source: str, evidence: str) -> int | None:
    needle = norm(evidence)
    for rank, r in enumerate(results, start=1):
        if r["source"] == source and needle in norm(r["content"]):
            return rank
    return None


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default=os.environ.get("RAG_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    p.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    p.add_argument("--modes", default="vector,hybrid,rerank", help="comma-separated")
    p.add_argument("-k", type=int, default=5, help="results per query (default 5)")
    p.add_argument("--json", type=Path, help="also write per-question results here")
    args = p.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    questions = load_questions(args.questions)
    client = httpx.Client(base_url=args.url, headers={"X-API-Key": args.api_key}, timeout=120)

    health = client.get("/health").json()
    print(
        f"Corpus: {health.get('documents')} documents, {health.get('chunks')} chunks | "
        f"embedder {health.get('embed_model')} | reranker {health.get('rerank_model')}"
    )
    print(f"Questions: {len(questions)} | top-k: {args.k}\n")

    ranks: dict[str, list[int | None]] = {m: [] for m in modes}
    latency: dict[str, list[float]] = {m: [] for m in modes}
    for q in questions:
        for mode in modes:
            started = time.perf_counter()
            r = client.post("/search", json={"query": q["question"], "mode": mode, "top_k": args.k})
            latency[mode].append((time.perf_counter() - started) * 1000)
            if r.status_code != 200:
                print(f"search failed ({r.status_code}): {r.text}", file=sys.stderr)
                return 1
            ranks[mode].append(first_hit_rank(r.json()["results"], q["source"], q["evidence"]))

    cutoffs = [c for c in (1, 3, 5) if c <= args.k] or [args.k]
    header = f"{'mode':<17}" + "".join(f"{'hit@' + str(c):>8}" for c in cutoffs)
    header += f"{'MRR':>8}{'median ms':>11}"
    print(header)
    print("-" * len(header))
    summary = {}
    for mode in modes:
        rs = ranks[mode]
        hit = {c: sum(1 for r in rs if r is not None and r <= c) / len(rs) for c in cutoffs}
        mrr = sum(1 / r for r in rs if r) / len(rs)
        ms = statistics.median(latency[mode])
        summary[mode] = {"hit": hit, "mrr": mrr, "median_ms": ms}
        line = f"{LABELS.get(mode, mode):<17}" + "".join(f"{hit[c]:>8.2f}" for c in cutoffs)
        print(line + f"{mrr:>8.3f}{ms:>11.0f}")

    print("\nRank of the first correct chunk per question (- = not in top k):\n")
    print(f"{'id':<5}" + "".join(f"{m:>8}" for m in modes) + "  question")
    for i, q in enumerate(questions):
        cells = "".join(f"{(ranks[m][i] or '-'):>8}" for m in modes)
        print(f"{q['id']:<5}{cells}  {q['question']}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "questions": [
                        {**q, "ranks": {m: ranks[m][i] for m in modes}}
                        for i, q in enumerate(questions)
                    ],
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
