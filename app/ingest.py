"""Ingest documents: parse, chunk, embed, and store in PostgreSQL.

Command line (inside the API container):

    python -m app.ingest samples            # a folder
    python -m app.ingest data/report.pdf    # or single files

Files are keyed by name. Re-ingesting an unchanged file is a no-op; a changed
file replaces its old chunks in one transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import db
from .chunking import chunk_text
from .config import get_settings
from .loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType, load
from .models import get_embedder

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source: str
    status: str  # "ingested", "unchanged" or "skipped"
    chunks: int = 0
    pages: int | None = None
    seconds: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def ingest_file(path: Path, source: str | None = None) -> IngestResult:
    started = time.perf_counter()
    s = get_settings()
    source = source or path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with db.pool().connection() as conn:
        row = conn.execute("SELECT sha256 FROM documents WHERE source = %s", (source,)).fetchone()
    if row and row[0] == digest:
        return IngestResult(source, "unchanged", detail="same content already ingested")

    try:
        doc = load(path)
    except UnsupportedFileType as exc:
        return IngestResult(source, "skipped", detail=str(exc))

    chunks: list[tuple[int | None, str]] = []
    for page in doc.pages:
        for text in chunk_text(page.text, s.chunk_words, s.chunk_overlap_words):
            chunks.append((page.number, text))
    if not chunks:
        return IngestResult(
            source, "skipped", detail="no extractable text (scanned PDFs need OCR first)"
        )

    vectors = get_embedder().embed_documents([text for _, text in chunks])
    page_count = len(doc.pages) if any(p.number for p in doc.pages) else None

    with db.pool().connection() as conn, conn.transaction():
        conn.execute("DELETE FROM documents WHERE source = %s", (source,))
        doc_id = conn.execute(
            "INSERT INTO documents (source, title, sha256, pages) VALUES (%s, %s, %s, %s) "
            "RETURNING id",
            (source, doc.title, digest, page_count),
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, page, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (doc_id, i, page, text, vec)
                    for i, ((page, text), vec) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )

    return IngestResult(
        source,
        "ingested",
        chunks=len(chunks),
        pages=page_count,
        seconds=round(time.perf_counter() - started, 2),
    )


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(
                f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF, DOCX, TXT and MD files.")
    parser.add_argument("paths", nargs="+", type=Path, help="files or folders")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    files = iter_files(args.paths)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    db.init(get_settings().database_url, get_embedder().dim)
    print(f"{'file':<40} {'status':<10} {'chunks':>6} {'pages':>5} {'secs':>6}")
    failed = 0
    for f in files:
        try:
            r = ingest_file(f)
        except Exception as exc:  # report it and carry on with the next file
            failed += 1
            print(f"{f.name:<40} {'error':<10} {exc}")
            continue
        pages = "" if r.pages is None else r.pages
        print(f"{r.source:<40} {r.status:<10} {r.chunks:>6} {pages:>5} {r.seconds:>6}")
        if r.detail and r.status == "skipped":
            print(f"  {r.detail}")
    db.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
