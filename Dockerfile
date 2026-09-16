# ─────────────────────────────────────────────────────────────────────────────
# MATZ AI Knowledge Assistant — API image
#
# Built for Render (or any container host). The app is a long-running FastAPI
# server, so it needs a container platform rather than a serverless one:
# it holds a ~90 MB embedding model in memory, and its upload endpoint accepts
# files far larger than a serverless request-body cap allows.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

# HF_HOME / SENTENCE_TRANSFORMERS_HOME are set so the model cache lands inside
# the image at build time (see the pre-download step below) instead of being
# fetched into a home directory on first request.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

# Build toolchain for any dependency without a prebuilt wheel. Removed in the
# same layer so it does not ship in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# The default PyPI torch wheel bundles CUDA libraries — roughly 2 GB of GPU
# code that is dead weight on a CPU-only host, and slow to build. The CPU
# index serves the same version at a fraction of the size.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Bake the embedding model into the image. Without this the first request
# after every deploy downloads ~90 MB, which is slow and fails outright if the
# host has no outbound access to Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Render supplies $PORT and it is not always 8000 — bind to it, not a literal.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
