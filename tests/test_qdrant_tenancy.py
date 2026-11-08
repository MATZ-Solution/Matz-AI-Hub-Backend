"""
tests/test_qdrant_tenancy.py — A4 part 2.

Done when: a code search finds no Qdrant call outside data/qdrant_store.py,
every store operation is scoped by org_filter, and the organization always
comes from the verified token, never from the client.

Run from Backend/:  .venv/bin/pytest tests/test_qdrant_tenancy.py -q
"""

import re
from pathlib import Path

import pytest
from qdrant_client.models import PointStruct

from fastapi.testclient import TestClient

from tests.conftest import session_headers
from api.app import app
from data import qdrant_store
from data.qdrant_store import TenantError

BACKEND = Path(__file__).resolve().parents[1]
STORE = BACKEND / "data" / "qdrant_store.py"

# Anything that talks to Qdrant directly.
QDRANT_CALL = re.compile(
    r"QdrantClient\(|\.query_points\(|\.scroll\(|\.upsert\(|points_selector|"
    r"create_payload_index|create_collection\(|recreate_collection|get_collection\(|"
    r"\.retrieve\(|\.set_payload\(|\.overwrite_payload\(|\.delete_payload\(|\.count\(collection_name"
)


def _python_sources():
    for path in BACKEND.rglob("*.py"):
        parts = set(path.relative_to(BACKEND).parts)
        if parts & {".venv", "tests", "__pycache__"} or path == STORE:
            continue
        yield path


def test_no_qdrant_call_outside_the_store():
    offenders = []
    for path in _python_sources():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            # Calls through the store are the allowed path; Supabase also has
            # .upsert(), so table calls are not Qdrant calls either.
            if QDRANT_CALL.search(code) and "qdrant_store." not in code and ".table(" not in code:
                offenders.append(f"{path.relative_to(BACKEND)}:{n}: {line.strip()}")
    assert offenders == [], "Qdrant used outside data/qdrant_store.py:\n" + "\n".join(offenders)


# ── org_filter and the store's guards ────────────────────────────────────────

class _Recorder:
    def __init__(self):
        self.calls = []

    def query_points(self, **kw):
        self.calls.append(("query", kw["query_filter"]))
        return type("R", (), {"points": []})()

    def scroll(self, **kw):
        self.calls.append(("scroll", kw["scroll_filter"]))
        return [], None

    def upsert(self, **kw):
        self.calls.append(("upsert", kw["points"]))

    def delete(self, **kw):
        self.calls.append(("delete", kw["points_selector"]))


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(qdrant_store, "_get_client", lambda: rec)
    return rec


def _org_of(flt):
    return [c.match.value for c in flt.must if c.key == "organization_id"]


def test_every_read_and_delete_carries_the_org(recorder):
    qdrant_store.query("org-a", [0.0], limit=3)
    qdrant_store.scroll_all("org-a", qdrant_store.match("collection_name", "HR"))
    qdrant_store.delete("org-a", qdrant_store.match("document_id", "d1"))
    for _, flt in recorder.calls:
        assert _org_of(flt) == ["org-a"]


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_missing_org_is_refused(recorder, bad):
    with pytest.raises(TenantError):
        qdrant_store.query(bad, [0.0])
    with pytest.raises(TenantError):
        qdrant_store.scroll_all(bad)
    assert recorder.calls == []


def test_org_cannot_be_overridden_by_a_condition():
    with pytest.raises(TenantError):
        qdrant_store.match("organization_id", "org-b")


def test_delete_without_a_condition_cannot_wipe_an_org(recorder):
    with pytest.raises(TenantError):
        qdrant_store.delete("org-a")
    assert recorder.calls == []


def test_upsert_refuses_points_of_another_org(recorder):
    mine = PointStruct(id=1, vector=[0.0], payload={"organization_id": "org-a"})
    theirs = PointStruct(id=2, vector=[0.0], payload={"organization_id": "org-b"})
    unlabelled = PointStruct(id=3, vector=[0.0], payload={})
    for bad in ([mine, theirs], [unlabelled]):
        with pytest.raises(TenantError):
            qdrant_store.upsert("org-a", bad)
    assert recorder.calls == []
    qdrant_store.upsert("org-a", [mine])
    assert recorder.calls[0][0] == "upsert"


# ── The organization comes from the token only ───────────────────────────────

def test_client_supplied_org_is_ignored(monkeypatch, recorder):
    client = TestClient(app, raise_server_exceptions=False, headers=session_headers(org="org-a"))
    # Query string, as the old API accepted.
    client.get("/documents?organization_id=org-b")
    # Body field, as ChatRequest/SearchRequest used to accept.
    monkeypatch.setattr("agents.langgraph_agent.embeddings.embeddings.embed_text", lambda t: [0.0])
    client.post("/search", json={"query": "leave", "organization_id": "org-b"})
    orgs = [o for _, flt in recorder.calls for o in _org_of(flt)]
    assert orgs and set(orgs) == {"org-a"}


def test_no_route_declares_an_organization_parameter():
    spec = app.openapi()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            names = [p["name"] for p in op.get("parameters", [])]
            assert "organization_id" not in names, f"{method.upper()} {path} accepts organization_id"
    for name, schema in spec["components"]["schemas"].items():
        if name.endswith(("Request", "Create", "Update")) or name.startswith("Body_"):
            assert "organization_id" not in schema.get("properties", {}), f"{name} accepts organization_id"
