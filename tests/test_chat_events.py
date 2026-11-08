"""
tests/test_chat_events.py — B2: every chat attempt is logged, failures included.

Run from Backend/:  .venv/bin/pytest tests/test_chat_events.py -q
"""

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.controllers import assistant_controller
from agents.langgraph_agent.LLMs import llm
from agents.langgraph_agent.LLMs.llm import ProviderError, UsageMeter, usage_meter

from tests.conftest import session_headers

client = TestClient(app, raise_server_exceptions=False, headers=session_headers())

CITATION = {
    "chunk_id": "c1", "document_id": "doc-hr", "document_title": "HR Policy",
    "collection_name": "HR", "page_number": 2, "cited_text": "20 days",
    "relevance_score": 0.8, "citation_order": 1,
}


@pytest.fixture
def events(monkeypatch):
    rows = []
    monkeypatch.setattr(assistant_controller, "record_chat_event", rows.append)
    return rows


def _agent(monkeypatch, fn):
    monkeypatch.setattr(assistant_controller, "run_agent_chat", fn)


def _ask(question="How many leave days?"):
    return client.post("/chat", json={"user_query": question})


def test_provider_error_is_logged(monkeypatch, events):
    def fail(**_):
        raise ProviderError("provider_unavailable", "401 expired key")
    _agent(monkeypatch, fail)
    assert _ask().status_code == 503
    assert len(events) == 1
    assert events[0]["outcome"] == "provider_error"
    assert events[0]["organization_id"] == "matz-demo-org"
    assert events[0]["question"] == "How many leave days?"
    assert events[0]["request_id"] not in (None, "-")


def test_rate_limited_and_timeout_are_logged(monkeypatch, events):
    def limited(**_):
        raise ProviderError("rate_limited", "429")
    _agent(monkeypatch, limited)
    _ask()
    monkeypatch.setattr(assistant_controller, "CHAT_TIMEOUT_SECONDS", 0.2)
    _agent(monkeypatch, lambda **_: time.sleep(1))
    _ask()
    assert [e["outcome"] for e in events] == ["rate_limited", "timeout"]


def test_unexpected_error_is_logged_as_internal(monkeypatch, events):
    def crash(**_):
        raise KeyError("boom")
    _agent(monkeypatch, crash)
    assert _ask().status_code == 500
    assert events[0]["outcome"] == "internal"


def test_answer_with_citation_is_answered(monkeypatch, events):
    _agent(monkeypatch, lambda **_: {
        "answer": "20 days [1]", "citations": [CITATION], "chat_history": [],
        "is_grounded": True, "target_collection": "HR",
    })
    res = _ask()
    assert res.status_code == 200
    e = events[0]
    assert e["outcome"] == "answered"
    assert e["cited_doc_ids"] == ["doc-hr"]
    assert e["collections_searched"] == ["HR"]
    assert e["latency_ms"] >= 0


def test_answer_without_citation_is_no_context(monkeypatch, events):
    _agent(monkeypatch, lambda **_: {
        "answer": "I couldn't find that.", "citations": [], "chat_history": [], "is_grounded": True,
    })
    assert _ask().status_code == 200
    assert events[0]["outcome"] == "no_context"


def test_event_logging_failure_never_breaks_chat(monkeypatch):
    from agents.langgraph_agent.utils import utils
    monkeypatch.setattr(utils, "get_supabase", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(assistant_controller, "record_chat_event", utils.record_chat_event)
    _agent(monkeypatch, lambda **_: {"answer": "hi", "citations": [], "chat_history": []})
    assert _ask().status_code == 200


# ── Token usage is summed across every provider call in the turn ────────────

def test_usage_meter_sums_tokens_across_threads(monkeypatch):
    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 3})()

    class Completions:
        def create(self, **kwargs):
            return type("R", (), {"usage": usage})()

    fake = type("C", (), {"chat": type("Ch", (), {"completions": Completions()})()})()
    monkeypatch.setattr(llm, "_get_client", lambda: fake)

    meter = UsageMeter()
    token = usage_meter.set(meter)
    try:
        llm._create_completion(model="big", messages=[])
        # Same pattern as the grader: worker threads started with copy_context.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in [pool.submit(contextvars.copy_context().run, llm._create_completion, model="small", messages=[])
                      for _ in range(2)]:
                f.result()
    finally:
        usage_meter.reset(token)

    assert meter.prompt_tokens == 30
    assert meter.completion_tokens == 9
    assert meter.models == ["big", "small"]
