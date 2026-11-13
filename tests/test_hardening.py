"""
tests/test_hardening.py — A6: security headers, CORS and error responses.

Done when: every listed header is sent, and a forced server error returns
only the error contract (no stack trace, no exception text).

Run from Backend/:  .venv/bin/pytest tests/test_hardening.py -q
"""

from fastapi.testclient import TestClient

from api.app import app
from api.controllers import assistant_controller
from api.security_headers import SECURITY_HEADERS
from tests.conftest import session_headers

client = TestClient(app, raise_server_exceptions=False, headers=session_headers())
anon = TestClient(app, raise_server_exceptions=False)


def _assert_hardened(res):
    for name in ("Content-Security-Policy", "X-Content-Type-Options", "Referrer-Policy"):
        assert res.headers.get(name) == SECURITY_HEADERS[name], name
    assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy"]


def test_headers_on_success_401_and_404():
    _assert_hardened(anon.get("/health"))
    _assert_hardened(anon.get("/documents"))          # 401
    _assert_hardened(client.get("/no-such-route"))    # 404


def test_forced_server_error_returns_only_the_contract(monkeypatch):
    def crash(**_):
        raise RuntimeError("Traceback: secret at /app/agents/x.py line 42")
    monkeypatch.setattr(assistant_controller, "run_agent_chat", crash)
    monkeypatch.setattr(assistant_controller, "record_chat_event", lambda row: None)
    res = client.post("/chat", json={"user_query": "hi"})
    assert res.status_code == 500
    _assert_hardened(res)
    body = res.json()
    assert set(body) == {"code", "message", "retryable", "request_id"}
    assert body["code"] == "internal"
    assert "Traceback" not in res.text and "secret" not in res.text and ".py" not in res.text


def test_cors_rejects_other_origins():
    res = anon.options("/documents", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in {k.lower() for k in res.headers}
    res = anon.get("/health", headers={"Origin": "https://evil.example"})
    assert res.headers.get("access-control-allow-origin") is None


def test_cors_origins_come_from_env(monkeypatch):
    import importlib
    from api.config import settings
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://app.example.com/, *, ")
    try:
        assert importlib.reload(settings).CORS_ALLOW_ORIGINS == ["https://app.example.com"]
    finally:
        monkeypatch.delenv("CORS_ALLOW_ORIGINS")
        importlib.reload(settings)
