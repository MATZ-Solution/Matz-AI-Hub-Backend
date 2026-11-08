"""
tests/test_auth.py — sign-in with email + password and database sessions.

Done when: every API route except /health and /auth/login returns 401
without a session; a wrong password shows "Email or password is incorrect".

Run from Backend/:  .venv/bin/pytest tests/test_auth.py -q
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeSupabase
from api import auth
from api.app import app
from api.auth import hash_password, hash_token, verify_password
from api.helpers import user_store

anon = TestClient(app, raise_server_exceptions=False)

PUBLIC = {"/health", "/auth/login"}


def _all_routes():
    # The OpenAPI schema lists every route, including those in included
    # routers (newer FastAPI wraps those, so app.routes is not flat).
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            yield method.upper(), path.replace("{", "").replace("}", "")


ROUTES = sorted(set(_all_routes()))


def test_route_list_is_not_empty():
    assert len(ROUTES) >= 15


@pytest.mark.parametrize("method,path", ROUTES)
def test_every_route_except_public_ones_requires_a_session(method, path):
    res = anon.request(method, path)
    if path in PUBLIC:
        assert res.status_code != 401 or path == "/auth/login"
    else:
        assert res.status_code == 401, f"{method} {path} is open"
        assert res.json()["code"] == "unauthorized"


def test_401_carries_cors_and_request_id():
    res = anon.get("/sessions", headers={"Origin": "https://example.com"})
    assert res.status_code == 401
    assert "x-request-id" in {k.lower() for k in res.headers}
    assert "access-control-allow-origin" in {k.lower() for k in res.headers}


# ── Real sign-in, session lookup and sign-out against a fake Supabase ────────

PASSWORD = "Correct-Horse-1"


@pytest.fixture
def db(monkeypatch):
    db = FakeSupabase()
    monkeypatch.setattr(user_store, "get_supabase", lambda: db)
    monkeypatch.setattr(auth, "_session_cache", {})
    user_store.create_user(email="admin@matz.com", name="Admin", password_hash=hash_password(PASSWORD),
                           role="admin", organization_id="matz-demo-org")
    return db


def _login(email="admin@matz.com", password=PASSWORD):
    return anon.post("/auth/login", json={"email": email, "password": password})


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_login_returns_a_session_that_works(db):
    res = _login("Admin@Matz.com")
    assert res.status_code == 200
    body = res.json()
    assert body["user"]["email"] == "admin@matz.com" and body["user"]["role"] == "admin"
    assert "password_hash" not in res.text
    me = anon.get("/me", headers=_bearer(body["token"]))
    assert me.status_code == 200 and me.json()["email"] == "admin@matz.com"


def test_only_the_token_hash_is_stored(db):
    token = _login().json()["token"]
    stored = db.rows("auth_sessions")
    assert len(stored) == 1
    assert stored[0]["token_hash"] == hash_token(token)
    assert token not in str(db.tables)


@pytest.mark.parametrize("email,password", [
    ("admin@matz.com", "wrong"),
    ("nobody@matz.com", PASSWORD),
    ("admin@matz.com", ""),
])
def test_wrong_credentials_show_one_message(db, email, password):
    res = _login(email, password)
    assert res.status_code == 401
    assert res.json()["message"] == "Email or password is incorrect"
    assert db.rows("auth_sessions") == []


def test_inactive_user_cannot_log_in_or_keep_using_a_session(db):
    token = _login().json()["token"]
    db.rows("users", email="admin@matz.com")[0]["is_active"] = False
    auth._session_cache.clear()
    assert anon.get("/me", headers=_bearer(token)).status_code == 401
    assert _login().status_code == 401


@pytest.mark.parametrize("token", ["", "garbage", "a" * 43, "Bearer"])
def test_unknown_tokens_are_rejected(db, token):
    assert anon.get("/me", headers=_bearer(token)).status_code == 401


def test_expired_session_is_rejected(db):
    token = _login().json()["token"]
    row = db.rows("auth_sessions")[0]
    row["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    auth._session_cache.clear()
    assert anon.get("/me", headers=_bearer(token)).status_code == 401


def test_expiry_is_enforced_even_from_the_cache(db, monkeypatch):
    token = _login().json()["token"]
    assert anon.get("/me", headers=_bearer(token)).status_code == 200  # now cached
    real_time = auth.time.time
    monkeypatch.setattr(auth.time, "time", lambda: real_time() + auth.SESSION_SECONDS + 5)
    db.rows("auth_sessions")[0]["expires_at"] = datetime.now(timezone.utc).isoformat()
    assert anon.get("/me", headers=_bearer(token)).status_code == 401


def test_logout_ends_the_session(db):
    token = _login().json()["token"]
    assert anon.post("/auth/logout", headers=_bearer(token)).status_code == 200
    assert db.rows("auth_sessions")[0]["revoked_at"]
    assert anon.get("/me", headers=_bearer(token)).status_code == 401


def test_password_change_signs_out_other_sessions_only(db):
    here = _login().json()["token"]
    other = _login().json()["token"]
    res = anon.post("/me/password", headers=_bearer(here),
                    json={"current_password": PASSWORD, "new_password": "New-password-22"})
    assert res.status_code == 200
    auth._session_cache.clear()
    assert anon.get("/me", headers=_bearer(here)).status_code == 200
    assert anon.get("/me", headers=_bearer(other)).status_code == 401
    assert _login(password="New-password-22").status_code == 200


def test_session_store_outage_fails_closed(db, monkeypatch):
    token = _login().json()["token"]
    auth._session_cache.clear()
    def down():
        raise ConnectionError("supabase unreachable")
    monkeypatch.setattr(user_store, "get_supabase", down)
    res = anon.get("/me", headers=_bearer(token))
    assert res.status_code == 500 and res.json()["code"] == "internal"


def test_passwords_are_argon2id():
    h = hash_password("x")
    assert h.startswith("$argon2id$")
    assert verify_password(h, "x") and not verify_password(h, "y")
    assert not verify_password(None, "x")
