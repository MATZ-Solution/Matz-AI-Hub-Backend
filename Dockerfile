FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so a container starts offline.
COPY agents/langgraph_agent/embeddings/embeddings.py /tmp/embeddings.py
RUN python -c "import importlib.util as u; s=u.spec_from_file_location('e','/tmp/embeddings.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.download_model()"

COPY . .

EXPOSE 8080
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-server-header"]
