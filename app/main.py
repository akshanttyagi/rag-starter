"""HTTP API: /ingest, /search, /ask, plus /health and /documents."""

from __future__ import annotations

import logging
import secrets
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Response, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__, db, llm, retrieval
from .config import get_settings
from .ingest import ingest_file
from .loaders import SUPPORTED_EXTENSIONS
from .models import get_embedder, get_reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    if not s.api_key:
        raise RuntimeError("API_KEY is not set. Set it in .env (see .env.example).")
    if s.api_key == "change-me":
        log.warning("API_KEY is the default 'change-me'. Set your own key in .env before sharing.")
    embedder = get_embedder()
    get_reranker()
    db.init(s.database_url, embedder.dim)
    threading.Thread(target=llm.watch_model, name="llm-watch", daemon=True).start()
    log.info("API started; /ask is available once %s serves %s", s.llm_provider, s.llm_model)
    yield
    db.close()


app = FastAPI(
    title="RAG Starter",
    version=__version__,
    description="Hybrid search (pgvector + PostgreSQL full-text + reranker) and cited answers "
    "from a self-hosted LLM.",
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="The value of the API_KEY setting."
)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = get_settings().api_key
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


protected = [Depends(require_api_key)]


# ---------- schemas ----------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["vector", "keyword", "hybrid", "rerank"] = "rerank"


class SearchHit(BaseModel):
    chunk_id: int
    source: str
    title: str
    page: int | None
    chunk_index: int
    content: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None


class SearchResponse(BaseModel):
    query: str
    mode: str
    took_ms: int
    results: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10, description="Passages given to the model.")


class Citation(BaseModel):
    ref: int = Field(description="The number used in the answer, e.g. [1].")
    cited: bool = Field(description="True if the answer refers to this source.")
    source: str
    title: str
    page: int | None
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    retrieval_ms: int
    generation_ms: int


# ---------- routes ----------


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"name": "rag-starter", "version": __version__, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(response: Response) -> dict:
    s = get_settings()
    database, documents, chunks = False, None, None
    try:
        with db.pool().connection() as conn:
            documents = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        database = True
    except Exception as exc:  # reported as database: false
        log.warning("health check: database error: %s", exc)
    ready = database and llm.status.ready
    response.status_code = 200 if ready else 503
    return {
        "status": "ok" if ready else "starting",
        "database": database,
        "documents": documents,
        "chunks": chunks,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model,
        "llm": {
            "provider": s.llm_provider,
            "model": s.llm_model,
            "ready": llm.status.ready,
            "detail": llm.status.detail,
        },
    }


@app.get("/documents", dependencies=protected)
def documents() -> list[dict]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT d.source, d.title, d.pages, count(c.id), d.created_at "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.source"
        ).fetchall()
    return [
        {"source": r[0], "title": r[1], "pages": r[2], "chunks": r[3], "ingested_at": r[4]}
        for r in rows
    ]


@app.delete("/documents/{source}", dependencies=protected)
def delete_document(source: str) -> dict:
    with db.pool().connection() as conn:
        deleted = conn.execute("DELETE FROM documents WHERE source = %s", (source,)).rowcount
    if not deleted:
        raise HTTPException(404, f"No document named {source!r}.")
    return {"deleted": source}


@app.post("/ingest", dependencies=protected)
def ingest(
    files: Annotated[list[UploadFile], File(description="PDF, DOCX, TXT or MD files")],
) -> dict:
    limit = get_settings().max_upload_mb * 1024 * 1024
    results = []
    for upload in files:
        name = Path(upload.filename or "upload").name
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            results.append({"source": name, "status": "skipped", "detail": "unsupported file type"})
            continue
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / name
            size = 0
            with dest.open("wb") as out:
                while block := upload.file.read(1024 * 1024):
                    size += len(block)
                    if size > limit:
                        raise HTTPException(413, f"{name} is larger than MAX_UPLOAD_MB")
                    out.write(block)
            results.append(ingest_file(dest, source=name).to_dict())
    return {"results": results}


@app.post("/search", dependencies=protected, response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    hits = retrieval.search(req.query, req.mode, req.top_k)
    return SearchResponse(
        query=req.query,
        mode=req.mode,
        took_ms=int((time.perf_counter() - started) * 1000),
        results=[SearchHit(**h.to_dict()) for h in hits],
    )


@app.post("/ask", dependencies=protected, response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    s = get_settings()
    if not llm.status.ready:
        raise HTTPException(503, f"LLM is not ready yet: {llm.status.detail}")

    started = time.perf_counter()
    hits = retrieval.search(req.question, "rerank", req.top_k)
    retrieved = time.perf_counter()
    if not hits:
        return AskResponse(
            question=req.question,
            answer=llm.NOT_FOUND,
            citations=[],
            model=s.llm_model,
            retrieval_ms=int((retrieved - started) * 1000),
            generation_ms=0,
        )

    passages = [(f"{h.source}, page {h.page}" if h.page else h.source, h.content) for h in hits]
    try:
        answer = llm.chat(llm.build_messages(req.question, passages))
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc
    generated = time.perf_counter()

    cited = set(llm.cited_numbers(answer, len(hits)))
    citations = [
        Citation(
            ref=i,
            cited=i in cited,
            source=h.source,
            title=h.title,
            page=h.page,
            chunk_id=h.chunk_id,
            score=h.score,
            text=h.content,
        )
        for i, h in enumerate(hits, start=1)
    ]
    return AskResponse(
        question=req.question,
        answer=answer,
        citations=citations,
        model=s.llm_model,
        retrieval_ms=int((retrieved - started) * 1000),
        generation_ms=int((generated - retrieved) * 1000),
    )
