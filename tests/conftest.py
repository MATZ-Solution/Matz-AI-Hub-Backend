"""Shared test setup.

AUTH_SECRET is fixed for tests *before* the app is imported, so tests never
depend on (or reveal) the real secret in .env — load_dotenv() does not
override variables that are already set.
"""

import os

os.environ["AUTH_SECRET"] = "test-only-auth-secret-not-used-anywhere-else"


def session_headers(org: str = "matz-demo-org", role: str = "admin", ttl: int = 3600) -> dict:
    from api.auth import make_token
    token = make_token(
        {"sub": "user-test-1", "email": "admin@test.local", "name": "Test Admin",
         "role": role, "org_id": org},
        purpose="session", ttl_seconds=ttl,
    )
    return {"Authorization": f"Bearer {token}"}
