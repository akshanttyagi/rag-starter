"""PostgreSQL access: schema setup and a shared connection pool.

One table holds documents, another holds their chunks. Each chunk carries both
search representations side by side:

* ``embedding``: a pgvector column with an HNSW index (semantic search)
* ``tsv``: a generated ``tsvector`` column with a GIN index (full-text search)
"""

from __future__ import annotations

import logging
import time

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    source      text NOT NULL UNIQUE,
    title       text NOT NULL,
    sha256      text NOT NULL,
    pages       integer,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  integer NOT NULL,
    page         integer,
    content      text NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding    vector({dim}) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def _connect_with_retry(url: str, attempts: int = 30) -> psycopg.Connection:
    for i in range(1, attempts + 1):
        try:
            return psycopg.connect(url, autocommit=True)
        except psycopg.OperationalError as exc:
            if i == attempts:
                raise
            log.info("database not ready (%s), retrying", str(exc).strip().splitlines()[0])
            time.sleep(2)
    raise RuntimeError("unreachable")


def init(database_url: str, embedding_dim: int) -> None:
    """Create the schema if needed, check the vector size, and open the pool."""
    global _pool
    with _connect_with_retry(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA.format(dim=int(embedding_dim)))
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        ).fetchone()
        if row and row[0] != embedding_dim:
            raise RuntimeError(
                f"The chunks table stores {row[0]}-dimensional vectors but the embedding "
                f"model produces {embedding_dim}. Re-create the database after changing "
                "EMBED_MODEL (for example: docker compose down -v), then ingest again."
            )
    if _pool is None:
        _pool = ConnectionPool(
            database_url, min_size=1, max_size=8, configure=register_vector, open=True
        )


def pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("database pool is not initialised; call db.init() first")
    return _pool


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
