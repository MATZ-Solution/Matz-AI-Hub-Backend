"""
tests/test_hybrid_retrieval.py — C2: hybrid retrieval.

Done when: the query "leave policy" returns the Annual Leave chunk first.

The corpus is the 3-page HR policy from test_chunking (contents page,
numbered sections) plus an IT policy, chunked by the real C1 chunker and
indexed in FakeQdrant, which scores dense queries by cosine and sparse ones
by BM25 with Qdrant's IDF formula.

Run from Backend/:  .venv/bin/pytest tests/test_hybrid_retrieval.py -q
"""

import pytest
from qdrant_client.models import SparseVector

from data import qdrant_store, retriever, sparse
from data.pipeline import ingestor
from data.pipeline.chunker import chunk_text
from tests.fakes import FakeQdrant, fake_vector
from tests.test_chunking import HR_PAGES

ORG = "org-hr"

IT_POLICY = """IT SECURITY POLICY
1. Passwords
Passwords must be at least 12 characters and changed every 90 days. This policy applies to all staff.
2. Devices
Laptops must be encrypted. Lost devices must be reported to IT within one hour; staff on leave remain responsible.
"""


def _index(qdrant, embed):
    hr_text = "\f".join("\n".join(lines) for lines in HR_PAGES)
    docs = [("doc-hr", "HR_Policy", "HR", hr_text, 3), ("doc-it", "IT_Security", "IT", IT_POLICY, 1)]
    for doc_id, title, collection, text, pages in docs:
        chunks = chunk_text(text, doc_id, pages, title=title)
        ingestor.ingest_chunks(chunks=chunks, document_id=doc_id, document_title=title,
                               collection_id=f"col-{collection}", collection_name=collection,
                               organization_id=ORG)


@pytest.fixture
def qdrant(monkeypatch):
    fake = FakeQdrant()
    monkeypatch.setattr(qdrant_store, "_get_client", lambda: fake)
    return fake


def topical_vector(text: str) -> list[float]:
    """A cheap stand-in embedding: hashed bag of words, so texts sharing words
    are closer. Deterministic and needs no model."""
    v = [0.0] * 384
    for term in sparse.terms(text):
        v[sparse.term_index(term) % 384] += 1.0
    return v if any(v) else fake_vector(text)


def _use_fake_embeddings(monkeypatch):
    monkeypatch.setattr(ingestor, "embed_texts", lambda texts: [topical_vector(t) for t in texts])
    monkeypatch.setattr("agents.langgraph_agent.embeddings.embeddings.embed_text", topical_vector)


def test_leave_policy_returns_annual_leave_first_bm25(qdrant, monkeypatch):
    """Runs in CI with no model: hashed bag-of-words embeddings plus BM25."""
    _use_fake_embeddings(monkeypatch)
    _index(qdrant, fake_vector)
    hits = retriever.hybrid_search(ORG, "leave policy")
    assert hits[0].payload["heading"] == "8. Annual Leave"
    assert len(hits) <= retriever.TOP_K


def _real_model():
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _real_model(), reason="embedding model not cached locally")
def test_leave_policy_returns_annual_leave_first_real_embeddings(qdrant):
    from agents.langgraph_agent.embeddings.embeddings import embed_texts
    _index(qdrant, None)
    hits = retriever.hybrid_search(ORG, "leave policy")
    # The top two are the Annual Leave section and its 8.1 subsection (they
    # score within 1% of each other); the contents page is never indexed.
    assert [h.payload["heading"].split(" > ")[0] for h in hits[:2]] == ["8. Annual Leave"] * 2
    assert hits[0].payload["page_number"] == 3
    assert hits[0].relevance >= retriever.RELEVANCE_FLOOR
    assert not any("Contents" in h.payload["content"] for h in hits)
    # Collection scope is a filter on the same search.
    it_only = retriever.hybrid_search(ORG, "leave policy", qdrant_store.match("collection_name", "IT"))
    assert all(h.payload["collection_name"] == "IT" for h in it_only)


def test_fusion_ranks_hits_found_by_both_searches_first():
    class P:
        def __init__(self, id, score=0.0, vector=None):
            self.id, self.score, self.payload, self.vector = id, score, {"id": id}, vector

    q = [1.0, 0.0]
    dense = [P("a", 0.9), P("b", 0.8), P("c", 0.7)]
    sparse_hits = [P("c", 5.0, {"dense": [1.0, 0.0]}), P("d", 4.0, {"dense": [0.0, 1.0]})]
    fused = retriever.fuse(dense, sparse_hits, q)
    assert [h.id for h in fused][:1] == ["c"]          # in both lists
    d = next(h for h in fused if h.id == "d")
    assert d.relevance == 0.0                          # sparse-only: cosine of its stored vector


def test_relevance_floor_drops_weak_matches(qdrant, monkeypatch):
    _use_fake_embeddings(monkeypatch)
    _index(qdrant, fake_vector)
    assert retriever.hybrid_search(ORG, "leave policy", floor=1.01) == []


def test_sparse_encoder():
    idx, val = sparse.encode_query("The Leave policies, LEAVE!")
    assert len(idx) == 2                               # stop word dropped, duplicates merged
    assert sparse.terms("policies leaves") == ["policy", "leave"]
    d_idx, d_val = sparse.encode_document("leave leave leave policy")
    weights = dict(zip(d_idx, d_val))
    assert weights[sparse.term_index("leave")] > weights[sparse.term_index("policy")]
    assert weights[sparse.term_index("leave")] < 3     # term frequency saturates
    assert sparse.encode_query("the of and") == ([], [])


def test_ingested_points_carry_both_vectors(qdrant, monkeypatch):
    _use_fake_embeddings(monkeypatch)
    _index(qdrant, fake_vector)
    for p in qdrant.points.values():
        assert set(p["vector"]) == {"dense", "bm25"}
        assert isinstance(p["vector"]["bm25"], SparseVector) and p["vector"]["bm25"].indices


def test_legacy_collection_is_copied_once_with_sparse_vectors(monkeypatch):
    from qdrant_client.models import PointStruct

    class Store(FakeQdrant):
        def __init__(self):
            super().__init__()
            self.collections = {"matz_chunks": {}}

        def get_collection(self, name):
            if name not in self.collections:
                raise ValueError("missing")

        def create_collection(self, collection_name, **_):
            self.collections[collection_name] = {}

        def upsert(self, collection_name, points):
            for p in points:
                self.collections[collection_name][p.id] = p

        def scroll(self, collection_name, limit=10, offset=None, with_payload=True, with_vectors=False, **_):
            pts = list(self.collections[collection_name].values())
            start = offset or 0
            page = pts[start:start + limit]
            return page, (start + limit if start + limit < len(pts) else None)

    store = Store()
    legacy = {"organization_id": "org-a", "content": "8. Annual Leave\n\n20 days", "document_id": "d"}
    store.collections["matz_chunks"][1] = PointStruct(id=1, vector=[0.1] * 384, payload=legacy)
    monkeypatch.setattr(qdrant_store, "_get_client", lambda: store)

    import logging
    qdrant_store.ensure_collection(logging.getLogger("t"))
    copied = store.collections[qdrant_store.COLLECTION][1]
    assert copied.payload == legacy
    assert copied.vector["dense"] == [0.1] * 384 and copied.vector["bm25"].indices
    qdrant_store.ensure_collection(logging.getLogger("t"))   # already exists: no-op
    assert len(store.collections[qdrant_store.COLLECTION]) == 1
