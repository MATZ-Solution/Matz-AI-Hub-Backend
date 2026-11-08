"""
tests/test_profile.py — B4: the signed-in user's profile and password.

Run from Backend/:  .venv/bin/pytest tests/test_profile.py -q
"""

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeSupabase
from api.app import app
from api.auth import hash_password, make_token, verify_password
from api.helpers import user_store


@pytest.fixture
def db(monkeypatch):
    db = FakeSupabase()
    monkeypatch.setattr(user_store, "get_supabase", lambda: db)
    return db


def _user(db, org="org-a", name="Mohid Ahmed", password="Old-password-1"):
    return user_store.create_user(email=f"{name.split()[0].lower()}@{org}.test", name=name,
                                  password_hash=hash_password(password), role="manager",
                                  organization_id=org)


def _client(row):
    token = make_token({"sub": row["id"], "email": row["email"], "name": row["name"],
                        "role": row["role"], "org_id": row["organization_id"]}, "session", 600)
    return TestClient(app, raise_server_exceptions=False, headers={"Authorization": f"Bearer {token}"})


def test_me_returns_the_database_profile(db):
    row = _user(db)
    res = _client(row).get("/me")
    assert res.status_code == 200
    assert res.json() == {"id": row["id"], "email": row["email"], "name": "Mohid Ahmed",
                          "role": "manager", "organization_id": "org-a"}
    assert "password_hash" not in res.text


def test_rename_is_saved_and_returned(db):
    row = _user(db)
    client = _client(row)
    assert client.patch("/me", json={"name": "  Sara Khan "}).json()["name"] == "Sara Khan"
    assert client.get("/me").json()["name"] == "Sara Khan"


@pytest.mark.parametrize("name", ["", "   ", "x" * 81])
def test_invalid_names_are_rejected(db, name):
    assert _client(_user(db)).patch("/me", json={"name": name}).status_code == 400


def test_change_password(db):
    row = _user(db)
    client = _client(row)
    res = client.post("/me/password", json={"current_password": "Old-password-1",
                                            "new_password": "New-password-22"})
    assert res.status_code == 200
    stored = db.rows("users", id=row["id"])[0]["password_hash"]
    assert verify_password(stored, "New-password-22") and not verify_password(stored, "Old-password-1")


@pytest.mark.parametrize("current,new,message", [
    ("wrong", "New-password-22", "Your current password is incorrect."),
    ("Old-password-1", "short", "Use a password between 10 and 128 characters."),
    ("Old-password-1", "Old-password-1", "Choose a password different from your current one."),
])
def test_bad_password_changes_are_refused(db, current, new, message):
    row = _user(db)
    res = _client(row).post("/me/password", json={"current_password": current, "new_password": new})
    assert res.status_code == 400 and res.json()["message"] == message
    assert verify_password(db.rows("users", id=row["id"])[0]["password_hash"], "Old-password-1")


def test_profile_cannot_reach_another_org(db):
    row = _user(db, org="org-a")
    forged = {**row, "organization_id": "org-b"}  # right user id, wrong org in token
    assert _client(forged).get("/me").status_code == 401
    assert _client(forged).patch("/me", json={"name": "Hijack"}).status_code == 401
    assert db.rows("users", id=row["id"])[0]["name"] == "Mohid Ahmed"


def test_deactivated_user_is_rejected(db):
    row = _user(db)
    db.rows("users", id=row["id"])[0]["is_active"] = False
    assert _client(row).get("/me").status_code == 401
