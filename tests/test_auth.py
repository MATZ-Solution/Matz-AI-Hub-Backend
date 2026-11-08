"""
tests/test_auth.py — A3: every API route except /health needs a valid token.

Run from Backend/:  .venv/bin/pytest tests/test_auth.py -q
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import session_headers
from api.app import app
from api.auth import hash_password, make_token, verify_password
from api.helpers import user_store

anon = TestClient(app, raise_server_exceptions=False)


def _all_routes():
    # The OpenAPI schema lists every route, including those in included
    # routers (newer FastAPI wraps those, so app.routes is not flat).
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            # Fill path params with a dummy value.
            yield method.upper(), path.replace("{", "").replace("}", "")


ROUTES = sorted(set(_all_routes()))


def test_route_list_is_not_empty():
    assert len(ROUTES) >= 15


@pytest.mark.parametrize("method,path", ROUTES)
def test_every_route_except_health_requires_a_token(method, path):
    res = anon.request(method, path)
    if path == "/health":
        assert res.status_code != 401
    else:
        assert res.status_code == 401, f"{method} {path} is open"
        assert res.json()["code"] == "unauthorized"


def test_garbage_expired_and_wrong_purpose_tokens_are_rejected():
    claims = {"sub": "u1", "email": "a@b.c", "role": "admin", "org_id": "org-a"}
    bad = [
        "not-a-jwt",
        make_token(claims, purpose="session", ttl_seconds=-120),   # expired
        make_token(claims, purpose="login", ttl_seconds=60),       # wrong purpose
        make_token({"sub": "u1"}, purpose="session", ttl_seconds=60),  # missing claims
    ]
    for token in bad:
        res = anon.get("/sessions", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 401


def test_token_signed_with_another_secret_is_rejected():
    import jwt, time
    forged = jwt.encode(
        {"sub": "u1", "email": "a@b.c", "role": "admin", "org_id": "org-a",
         "purpose": "session", "iat": int(time.time()), "exp": int(time.time()) + 60},
        "attacker-secret", algorithm="HS256",
    )
    assert anon.get("/sessions", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_valid_session_token_is_accepted(monkeypatch):
    from api.controllers import assistant_controller
    monkeypatch.setattr(assistant_controller, "get_sessions", lambda org: [])
    res = anon.get("/sessions", headers=session_headers())
    assert res.status_code == 200


def test_401_carries_cors_and_request_id():
    res = anon.get("/sessions", headers={"Origin": "https://example.com"})
    assert res.status_code == 401
    assert "x-request-id" in {k.lower() for k in res.headers}
    assert "access-control-allow-origin" in {k.lower() for k in res.headers}


# ── POST /auth/login ─────────────────────────────────────────────────────────



def _login(email, password):
    token = make_token({}, purpose="login", ttl_seconds=60)
    return anon.post("/auth/login", json={"email": email, "password": password},
                     headers={"Authorization": f"Bearer {token}"})


@pytest.fixture
def one_user(monkeypatch):
    user = {
        "id": "11111111-1111-1111-1111-111111111111", "email": "admin@matz.com",
        "name": "Admin", "role": "admin", "organization_id": "matz-demo-org",
        "is_active": True, "password_hash": hash_password("Correct-Horse-1"),
    }
    monkeypatch.setattr(user_store, "get_user_by_email",
                        lambda email: user if email.strip().lower() == user["email"] else None)
    monkeypatch.setattr(user_store, "record_login", lambda user_id: None)
    return user


def test_login_success_returns_user_without_hash(one_user):
    res = _login("Admin@Matz.com", "Correct-Horse-1")
    assert res.status_code == 200
    body = res.json()
    assert body["email"] == "admin@matz.com" and body["role"] == "admin"
    assert body["organization_id"] == "matz-demo-org"
    assert "password_hash" not in res.text


@pytest.mark.parametrize("email,password", [
    ("admin@matz.com", "wrong"),
    ("nobody@matz.com", "Correct-Horse-1"),
])
def test_wrong_credentials_show_one_message(one_user, email, password):
    res = _login(email, password)
    assert res.status_code == 401
    assert res.json()["message"] == "Email or password is incorrect"


def test_inactive_user_cannot_log_in(one_user):
    one_user["is_active"] = False
    assert _login("admin@matz.com", "Correct-Horse-1").status_code == 401


def test_login_endpoint_needs_the_login_token(one_user):
    res = anon.post("/auth/login", json={"email": "admin@matz.com", "password": "Correct-Horse-1"})
    assert res.status_code == 401
    # A session token is not a login token.
    res = anon.post("/auth/login", json={"email": "admin@matz.com", "password": "Correct-Horse-1"},
                    headers=session_headers())
    assert res.status_code == 401


def test_passwords_are_argon2id():
    h = hash_password("x")
    assert h.startswith("$argon2id$")
    assert verify_password(h, "x") and not verify_password(h, "y")
    assert not verify_password(None, "x")
