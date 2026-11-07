# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models

WORKDIR /app

# CPU-only PyTorch: the embedding and reranker models are small, and the LLM
# runs in its own container (Ollama or vLLM).
RUN pip install --retries 10 --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY requirements.txt .
RUN pip install --retries 10 -r requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /opt/models \
    && chown app:app /opt/models
USER app

# Bake the models into the image so the API never needs the Hugging Face Hub at runtime.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=BAAI/bge-reranker-base
RUN python -c "import sys; from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(sys.argv[1]); CrossEncoder(sys.argv[2])" "$EMBED_MODEL" "$RERANK_MODEL"

COPY --chown=app:app app ./app
COPY --chown=app:app eval ./eval
COPY --chown=app:app samples ./samples

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    EMBED_MODEL=${EMBED_MODEL} \
    RERANK_MODEL=${RERANK_MODEL}

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
