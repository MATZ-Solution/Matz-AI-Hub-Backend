"""
tests/test_tenancy.py — F1: tenant-isolation suite with two organizations.

Org A and org B each get documents, a collection, a chat session and
settings, created through the real code paths (ingestor, Supabase helpers)
against in-memory fakes. Then org B tries everything the task list names:
a forged org id in body, query and headers; guessing org A's ids on read,
delete, replace and citation; and chat retrieval across orgs.

Done when: org B can never read, change or cite org A's data.
Run from Backend/:  .venv/bin/pytest tests/test_tenancy.py -q
"""

import io

import pytest
from fastapi.testclient import TestClient

from tests.conftest import session_headers
from tests.fakes import FakeQdrant, FakeSupabase, fake_vector

from api.app import app
from api.helpers import user_store
from agents.langgraph_agent.utils import utils
from agents.langgraph_agent.nodes import node
from agents.langgraph_agent.tools.tools import build_citations
from data import qdrant_store
from data.pipeline import ingestor

ORG_A, ORG_B = "org-a", "org-b"
DOC_A, DOC_B = "doc-a-hr", "doc-b-it"
SECRET_A = "Org A secret: annual leave is 14 days"


@pytest.fixture
def world(monkeypatch):
    """Two fully populated organizations on in-memory Qdrant and Supabase."""
    qdrant, db = FakeQdrant(), FakeSupabase()
    monkeypatch.setattr(qdrant_store, "_get_client", lambda: qdrant)
    monkeypatch.setattr(utils, "get_supabase", lambda: db)
    monkeypatch.setattr(user_store, "get_supabase", lambda: db)
    monkeypatch.setattr(ingestor, "embed_texts", lambda texts: [fake_vector(t) for t in texts])
    monkeypatch.setattr(node, "embed_text", fake_vector)
    monkeypatch.setattr(node, "correct_query", lambda text: text)
    monkeypatch.setattr("agents.langgraph_agent.embeddings.embeddings.embed_text", fake_vector)

    def seed(org, doc, title, collection, text):
        ingestor.ingest_chunks(
            chunks=[{"chunk_index": i, "content": f"{text} (part {i})", "page_number": i + 1,
                     "document_id": doc, "token_count": 10} for i in range(3)],
            document_id=doc, document_title=title, collection_id=f"col-{collection}",
            collection_name=collection, organization_id=org,
        )
        collection_row = utils.create_collection_in_db(collection, "desc", "folder", organization_id=org)
        session_id = utils.create_session(org)
        utils.save_message(session_id, "user", f"question from {org}", [])
        utils.save_message(session_id, "assistant", text, [], is_grounded=True, response_time_ms=100)
        utils.upsert_workspace_settings(org, f"Workspace {org}")
        return {"doc": doc, "collection_id": collection_row["id"], "session_id": session_id}

    a = seed(ORG_A, DOC_A, "A HR Policy", "HR", SECRET_A)
    b = seed(ORG_B, DOC_B, "B IT Policy", "IT", "Org B text")
    return {"qdrant": qdrant, "db": db, "a": a, "b": b}


@pytest.fixture
def as_b():
    return TestClient(app, raise_server_exceptions=False, headers=session_headers(org=ORG_B))


def _a_chunks(world):
    return world["qdrant"].payloads(organization_id=ORG_A)


def _assert_no_org_a_data(response):
    assert SECRET_A not in response.text
    assert DOC_A not in response.text
    assert "A HR Policy" not in response.text


# ── Forged organization id: body, query string, headers ─────────────────────

FORGERIES = [
    {"params": {"organization_id": ORG_A}},
    {"headers": {"X-Organization-Id": ORG_A, "Organization-Id": ORG_A, "X-Org-Id": ORG_A}},
]


@pytest.mark.parametrize("forgery", FORGERIES)
@pytest.mark.parametrize("path", ["/documents", "/collections", "/stats", "/sessions",
                                  "/settings/workspace", "/settings/usage", "/analytics/summary"])
def test_forged_org_on_reads_is_ignored(world, as_b, path, forgery):
    res = as_b.get(path, **forgery)
    assert res.status_code == 200
    _assert_no_org_a_data(res)
    assert "Workspace org-a" not in res.text


def test_forged_org_in_search_body(world, as_b):
    res = as_b.post("/search", json={"query": "annual leave", "organization_id": ORG_A})
    assert res.status_code == 200
    _assert_no_org_a_data(res)
    assert all(r["document_id"] == DOC_B for r in res.json()["results"])


def test_forged_org_in_settings_body_changes_only_own_org(world, as_b):
    res = as_b.put("/settings/workspace", json={"name": "Hijacked", "organization_id": ORG_A})
    assert res.status_code == 200
    assert world["db"].rows("workspace_settings", organization_id=ORG_A)[0]["name"] == "Workspace org-a"
    assert world["db"].rows("workspace_settings", organization_id=ORG_B)[0]["name"] == "Hijacked"


def test_forged_org_in_collection_create_lands_in_own_org(world, as_b):
    as_b.post("/collections", json={"name": "Planted", "organization_id": ORG_A})
    assert not world["db"].rows("collections", organization_id=ORG_A, name="Planted")
    assert world["db"].rows("collections", organization_id=ORG_B, name="Planted")


def test_forged_org_in_upload_form_lands_in_own_org(world, as_b, monkeypatch):
    from api.controllers import knowledge_controller
    monkeypatch.setattr(knowledge_controller, "extract_text",
                        lambda path: {"success": True, "text": "uploaded text. " * 20, "page_count": 1, "method": "txt"})
    res = as_b.post("/ingest", files={"file": ("x.txt", io.BytesIO(b"hello"), "text/plain")}, data={
        "document_title": "Upload", "collection_name": "IT", "collection_id": "col-IT",
        "document_id": "doc-new", "organization_id": ORG_A,
    })
    assert res.status_code == 200
    assert not world["qdrant"].payloads(organization_id=ORG_A, document_id="doc-new")
    assert world["qdrant"].payloads(organization_id=ORG_B, document_id="doc-new")


# ── Guessing org A's ids ─────────────────────────────────────────────────────

def test_guess_document_id_on_read(world, as_b):
    for res in (as_b.get("/documents"),
                as_b.post("/search", json={"query": "leave"}),
                as_b.post("/search", json={"query": "leave", "collection_name": "HR"})):
        _assert_no_org_a_data(res)


def test_guess_document_id_on_delete(world, as_b):
    before = len(_a_chunks(world))
    res = as_b.delete(f"/documents/{DOC_A}")
    assert res.status_code == 404
    assert len(_a_chunks(world)) == before == 3


def test_guess_document_id_on_replace(world, as_b):
    before = [p["content"] for p in _a_chunks(world)]
    res = as_b.put(f"/documents/{DOC_A}",
                   files={"file": ("evil.txt", io.BytesIO(b"overwrite"), "text/plain")},
                   data={"document_title": "Evil", "collection_name": "HR", "collection_id": "col-HR"})
    assert res.status_code == 404
    assert [p["content"] for p in _a_chunks(world)] == before
    assert not world["qdrant"].payloads(document_id=DOC_A, organization_id=ORG_B)


def test_guess_session_id(world, as_b):
    sid = world["a"]["session_id"]
    assert as_b.get(f"/sessions/{sid}/messages").status_code == 404
    assert as_b.delete(f"/sessions/{sid}").status_code == 404
    assert world["db"].rows("sessions", id=sid), "org A's session was deleted"
    res = as_b.post("/chat", json={"user_query": "hi", "session_id": sid})
    assert res.status_code == 404
    assert len(world["db"].rows("messages", session_id=sid)) == 2, "org B wrote into org A's chat"


def test_guess_collection_id(world, as_b):
    cid = world["a"]["collection_id"]
    assert as_b.delete(f"/collections/{cid}").status_code == 404
    assert world["db"].rows("collections", id=cid)


# ── Chat retrieval and citations across orgs ─────────────────────────────────

def _state(org):
    return {"user_query": "how many leave days?", "organization_id": org, "chat_history": []}


def test_chat_retrieval_never_crosses_orgs(world):
    for retrieve in (node.retrieve_context, node.retrieve_all):
        docs = retrieve(_state(ORG_B))["retrieved_docs"]
        assert docs, "org B should still find its own chunks"
        assert not any(SECRET_A in d or "A HR Policy" in d for d in docs)


def test_citations_never_name_another_orgs_document(world):
    state = node.retrieve_all(_state(ORG_B))
    citations = build_citations(state)["citations"]
    assert citations
    assert {c["document_title"] for c in citations} == {"B IT Policy"}


def test_org_a_still_sees_its_own_data(world):
    as_a = TestClient(app, raise_server_exceptions=False, headers=session_headers(org=ORG_A))
    docs = as_a.get("/documents").json()["documents"]
    assert [d["document_id"] for d in docs] == [DOC_A]
    assert as_a.get(f"/sessions/{world['a']['session_id']}/messages").status_code == 200
