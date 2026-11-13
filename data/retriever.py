"""
data/retriever.py
------------------
Hybrid retrieval (task C2), shared by chat and Content search.

  1. Dense (embedding) and sparse (BM25) search, top 20 each, in Qdrant.
  2. Reciprocal rank fusion: score = Σ 1 / (60 + rank) over both lists.
  3. Relevance floor on the dense cosine similarity, so a chunk that only
     shares a common word with the question is not cited.
  4. Keep the top 5.

RELEVANCE_FLOOR is provisional until the C5 benchmark sets it.
"""

import math
from dataclasses import dataclass

from data import qdrant_store

CANDIDATES = 20
TOP_K = 5
RRF_K = 60
RELEVANCE_FLOOR = 0.2


@dataclass
class Hit:
    id: object
    payload: dict
    relevance: float   # dense cosine similarity to the question, 0-1
    fused: float       # reciprocal-rank-fusion score; the ranking key


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _dense_vector(point) -> list[float] | None:
    vector = getattr(point, "vector", None)
    if isinstance(vector, dict):
        return vector.get(qdrant_store.DENSE)
    return vector


def fuse(dense_hits: list, sparse_hits: list, query_vector: list[float], k: int = RRF_K) -> list[Hit]:
    """Both ranked lists fused by RRF, best first. Every hit gets a dense
    relevance: its query score when it came from the dense search, else the
    cosine of its stored dense vector."""
    hits: dict = {}
    for rank, point in enumerate(dense_hits, 1):
        hits[point.id] = Hit(point.id, point.payload, float(point.score), 1 / (k + rank))
    for rank, point in enumerate(sparse_hits, 1):
        if point.id in hits:
            hits[point.id].fused += 1 / (k + rank)
            continue
        vector = _dense_vector(point)
        relevance = _cosine(query_vector, vector) if vector else 0.0
        hits[point.id] = Hit(point.id, point.payload, relevance, 1 / (k + rank))
    return sorted(hits.values(), key=lambda h: (h.fused, h.relevance), reverse=True)


def hybrid_search(org_id: str, query: str, *conditions, top_k: int = TOP_K,
             floor: float = RELEVANCE_FLOOR) -> list[Hit]:
    """The org's best `top_k` chunks for `query`, passing the relevance floor."""
    from agents.langgraph_agent.embeddings import embeddings  # patched in tests
    vector = embeddings.embed_text(query)
    dense_hits, sparse_hits = qdrant_store.hybrid_query(org_id, vector, query, *conditions, limit=CANDIDATES)
    return [h for h in fuse(dense_hits, sparse_hits, vector) if h.relevance >= floor][:top_k]
