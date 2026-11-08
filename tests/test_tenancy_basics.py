"""
tests/test_tenancy_basics.py — A4 part 1: the cross-tenant leaks found in the
code review, before auth exists. The full two-org suite is task F1.

Run from Backend/:  .venv/bin/pytest tests/test_tenancy_basics.py -q
"""

import uuid

from fastapi.testclient import TestClient

from api.app import app
from agents.langgraph_agent.nodes import node
from agents.langgraph_agent.utils import utils

from tests.conftest import session_headers

client = TestClient(app, raise_server_exceptions=False, headers=session_headers())


# ── Chat retrieval uses the request's organization ──────────────────────────

class _RecordingQdrant:
    def __init__(self):
        self.filters = []

    def query_points(self, query_filter=None, **_):
        self.filters.append(query_filter)
        return type("R", (), {"points": []})()

    def scroll(self, scroll_filter=None, **_):
        self.filters.append(scroll_filter)
        return [], None


def _org_values(filters):
    return [c.match.value for f in filters for c in f.must if c.key == "organization_id"]


def _state(org):
    return {"user_query": "leave policy", "organization_id": org, "chat_history": []}


def test_retrieve_context_filters_on_request_org(monkeypatch):
    fake = _RecordingQdrant()
    monkeypatch.setattr(node, "_get_client", lambda: fake)
    monkeypatch.setattr(node, "embed_text", lambda text: [0.0] * 384)
    monkeypatch.setattr(node, "correct_query", lambda text: text)
    node.retrieve_context(_state("org-b"))
    assert _org_values(fake.filters) == ["org-b"]


def test_retrieve_all_filters_on_request_org(monkeypatch):
    fake = _RecordingQdrant()
    monkeypatch.setattr(node, "_get_client", lambda: fake)
    node.retrieve_all(_state("org-b"))
    assert _org_values(fake.filters) == ["org-b"]


def test_no_hardcoded_org_left_in_retrieval():
    import inspect
    source = inspect.getsource(node)
    assert "matz-demo-org" not in source
    assert "ORGANIZATION_ID" not in source


# ── Supabase deletes and reads are scoped to the organization ───────────────

class _RecordingQuery:
    """Stands in for supabase-py's query builder; records every .eq()."""

    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __getattr__(self, name):
        def chain(*args, **kwargs):
            self.log.append((name, args))
            return self
        return chain

    def execute(self):
        return type("Res", (), {"data": self.rows})()


def _fake_supabase(monkeypatch, rows):
    log = []
    fake = type("S", (), {"table": lambda self, name: _RecordingQuery(log, rows)})()
    monkeypatch.setattr(utils, "get_supabase", lambda: fake)
    monkeypatch.setattr(utils, "_exec", lambda q: q.execute())
    return log


def test_delete_session_is_scoped_to_org(monkeypatch):
    log = _fake_supabase(monkeypatch, rows=[])
    sid = str(uuid.uuid4())
    assert utils.delete_session(sid, "org-b") is False
    assert ("eq", ("organization_id", "org-b")) in log


def test_delete_collection_is_scoped_to_org(monkeypatch):
    log = _fake_supabase(monkeypatch, rows=[])
    assert utils.delete_collection_from_db(str(uuid.uuid4()), "org-b") is False
    assert ("eq", ("organization_id", "org-b")) in log


def test_get_messages_refuses_other_orgs_session(monkeypatch):
    _fake_supabase(monkeypatch, rows=[])  # ownership lookup finds nothing
    assert utils.get_messages(str(uuid.uuid4()), "org-b") is None


def test_non_uuid_ids_never_reach_the_database(monkeypatch):
    log = _fake_supabase(monkeypatch, rows=[{"id": "x"}])
    assert utils.delete_session("../../etc", "org-a") is False
    assert utils.session_belongs_to_org("not-a-uuid", "org-a") is False
    assert log == []


# ── API: another org's ids come back as not_found, never as success ─────────

def test_api_delete_other_orgs_session_is_not_found(monkeypatch):
    from api.controllers import assistant_controller
    monkeypatch.setattr(assistant_controller, "delete_session", lambda sid, org: False)
    res = client.delete(f"/sessions/{uuid.uuid4()}")
    assert res.status_code == 404
    assert res.json()["code"] == "not_found"


def test_api_read_other_orgs_messages_is_not_found(monkeypatch):
    from api.controllers import assistant_controller
    monkeypatch.setattr(assistant_controller, "get_messages", lambda sid, org: None)
    res = client.get(f"/sessions/{uuid.uuid4()}/messages")
    assert res.status_code == 404


def test_chat_into_other_orgs_session_is_refused(monkeypatch):
    from api.controllers import assistant_controller
    monkeypatch.setattr(assistant_controller, "session_belongs_to_org", lambda sid, org: False)
    ran = []
    monkeypatch.setattr(assistant_controller, "run_agent_chat", lambda **kw: ran.append(kw))
    res = client.post("/chat", json={"user_query": "hi", "session_id": str(uuid.uuid4())})
    assert res.status_code == 404
    assert ran == []  # the agent never ran, so nothing was written
