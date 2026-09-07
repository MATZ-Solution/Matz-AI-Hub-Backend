"""
src/models/embeddings.py
-------------------------
Single place for all embedding logic.
Both the KB upload script and the retrieve node import from here —
so model name and loading behaviour are never duplicated.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast, free, runs locally (no API key needed)
  - Good enough for semantic search on short text chunks

Usage:
    from agents.langgraph_agent.embeddings.embeddings import get_embedding_model, embed_text, embed_texts
"""

from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIMENSIONS = 384

# Lazy singleton — loaded once per process, reused across all calls
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance (loads on first call)."""
    global _model
    if _model is None:
        try:
            # Use local cache first — fast, no internet needed
            _model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        except Exception:
            # First time on a new server — download and cache it
            _model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=False)
    return _model

def embed_text(text: str) -> list[float]:
    """
    Embed a single string and return a float list.
    Used by retrieve.py for query embedding.
    """
    model = get_embedding_model()
    return model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of strings and return a list of float lists.
    Used by upload_to_qdrant.py for bulk document embedding.
    Batching is faster than calling embed_text() in a loop.
    """
    model = get_embedding_model()
    return model.encode(texts).tolist()
