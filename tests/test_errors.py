"""
tests/test_errors.py — A1 (health, startup check) and A2 (error contract).

Run from Backend/:  .venv/bin/pytest tests/test_errors.py -q

No network: the agent, provider and Qdrant are patched. TestClient is used
without a `with` block, so the startup checks (which need real secrets) only
run in the test that targets them.
"""

import httpx
import groq
import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.config import settings
from agents.langgraph_agent.LLMs import llm
from agents.langgraph_agent.LLMs.llm import ProviderError
from api.controllers import assistant_controller
from api.routes import health_routes

CONTRACT_KEYS = {"code", "message", "retryable", "request_id"}
RAW_PROVIDER_TEXT = "Error code: 401 - {'error': {'message': 'Invalid API Key', 'code': 'expired_api_key'}}"

client = TestClient(app, raise_server_exceptions=False)


def _chat(monkeypatch, exc):
    def boom(**_):
        raise exc
    monkeypatch.setattr(assistant_controller, "run_agent_chat", boom)
    return client.post("/chat", json={"user_query": "How many paid annual leave days?"})


# ── A2: provider failure shows only the friendly sentence ────────────────────

def test_expired_key_returns_friendly_message_only(monkeypatch):
    res = _chat(monkeypatch, ProviderError("provider_unavailable", RAW_PROVIDER_TEXT))
    body = res.json()
    assert res.status_code == 503
    assert set(body) == CONTRACT_KEYS
    assert body["code"] == "provider_unavailable"
    assert body["message"] == "The assistant is unavailable right now. Try again in a moment."
    assert body["retryable"] is True
    for leak in ("Invalid API Key", "expired_api_key", "401", "Traceback"):
        assert leak not in res.text


def test_provider_rate_limit_maps_to_rate_limited(monkeypatch):
    res = _chat(monkeypatch, ProviderError("rate_limited", "429 slow down"))
    assert res.status_code == 429
    assert res.json()["code"] == "rate_limited"


def test_unexpected_exception_returns_internal_without_details(monkeypatch):
    res = _chat(monkeypatch, ValueError("secret internal detail /etc/passwd"))
    body = res.json()
    assert res.status_code == 500
    assert set(body) == CONTRACT_KEYS
    assert body["code"] == "internal"
    assert "secret internal detail" not in res.text


def test_agent_timeout_returns_timeout(monkeypatch):
    import time
    monkeypatch.setattr(assistant_controller, "CHAT_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(assistant_controller, "run_agent_chat", lambda **_: time.sleep(1))
    res = client.post("/chat", json={"user_query": "slow question"})
    assert res.status_code == 504
    assert res.json()["code"] == "timeout"


def test_validation_error_uses_contract():
    res = client.post("/chat", json={})
    assert res.status_code == 400
    assert set(res.json()) == CONTRACT_KEYS
    assert res.json()["code"] == "invalid_request"


def test_request_id_is_echoed_and_in_body(monkeypatch):
    res = client.post(
        "/chat", json={},
        headers={"X-Request-ID": "test-request-1234"},
    )
    assert res.headers["X-Request-ID"] == "test-request-1234"
    assert res.json()["request_id"] == "test-request-1234"


def test_error_responses_keep_cors_headers(monkeypatch):
    # An unhandled error must still carry CORS headers, or the browser hides
    # the contract body and reports a network error instead.
    def boom(**_):
        raise ValueError("boom")
    monkeypatch.setattr(assistant_controller, "run_agent_chat", boom)
    res = client.post("/chat", json={"user_query": "x"}, headers={"Origin": "https://example.com"})
    assert res.status_code == 500
    assert "access-control-allow-origin" in {k.lower() for k in res.headers}


# ── A2: provider retry schedule ──────────────────────────────────────────────

def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("provider said no", response=httpx.Response(status, request=request), body=None)


class _FakeCompletions:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def create(self, **_):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _fake_client(monkeypatch, errors):
    completions = _FakeCompletions(errors)
    fake = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return completions, sleeps


def test_retries_429_and_5xx_twice_after_1s_then_3s(monkeypatch):
    completions, sleeps = _fake_client(monkeypatch, [
        _status_error(groq.RateLimitError, 429),
        _status_error(groq.InternalServerError, 503),
    ])
    assert llm._create_completion(model="m", messages=[]) == "ok"
    assert completions.calls == 3
    assert sleeps == [1, 3]


def test_gives_up_after_two_retries(monkeypatch):
    completions, sleeps = _fake_client(monkeypatch, [_status_error(groq.InternalServerError, 500)] * 3)
    with pytest.raises(ProviderError) as info:
        llm._create_completion(model="m", messages=[])
    assert info.value.kind == "provider_unavailable"
    assert completions.calls == 3


def test_expired_key_is_not_retried(monkeypatch):
    completions, sleeps = _fake_client(monkeypatch, [_status_error(groq.AuthenticationError, 401)])
    with pytest.raises(ProviderError) as info:
        llm._create_completion(model="m", messages=[])
    assert info.value.kind == "provider_unavailable"
    assert completions.calls == 1 and sleeps == []


# ── A1: /health and the startup check ────────────────────────────────────────

@pytest.fixture
def fresh_health_cache(monkeypatch):
    monkeypatch.setattr(health_routes, "_provider_cache", None)


def test_health_green(monkeypatch, fresh_health_cache):
    monkeypatch.setattr(llm, "ping_provider", lambda model: None)
    monkeypatch.setattr(health_routes, "_check_qdrant", lambda: True)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "provider": True, "qdrant": True}


def test_health_red_when_provider_fails(monkeypatch, fresh_health_cache):
    def expired(model):
        raise ProviderError("provider_unavailable", RAW_PROVIDER_TEXT)
    monkeypatch.setattr(llm, "ping_provider", expired)
    monkeypatch.setattr(health_routes, "_check_qdrant", lambda: True)
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json() == {"ok": False, "provider": False, "qdrant": True}
    assert "expired_api_key" not in res.text


def test_health_caches_provider_result(monkeypatch, fresh_health_cache):
    calls = []
    monkeypatch.setattr(llm, "ping_provider", lambda model: calls.append(model))
    monkeypatch.setattr(health_routes, "_check_qdrant", lambda: True)
    client.get("/health")
    client.get("/health")
    assert len(calls) == 1


def test_api_refuses_to_start_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert "GROQ_API_KEY" in settings.missing_required_env()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        with TestClient(app):
            pass
