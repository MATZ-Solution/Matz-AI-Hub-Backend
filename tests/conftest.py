"""Shared test setup.

Most tests only need "a signed-in user of org X with role Y". Real sessions
live in Supabase, so for those tests conftest wraps api.auth.lookup_session:
tokens made by session_headers() resolve to a test user, and every other
token goes through the real lookup. This exists only in the test process —
production code has no such token format. tests/test_auth.py exercises the
real sign-in, session lookup, expiry and sign-out end to end.
"""

import json

import api.auth as auth

TEST_PREFIX = "test-session:"
_real_lookup = auth.lookup_session


def _lookup_with_test_tokens(token: str):
    if token.startswith(TEST_PREFIX):
        data = json.loads(token[len(TEST_PREFIX):])
        return auth.CurrentUser(**data)
    return _real_lookup(token)


auth.lookup_session = _lookup_with_test_tokens


def session_headers(org: str = "matz-demo-org", role: str = "admin", user_id: str = "user-test-1") -> dict:
    user = {"id": user_id, "email": "admin@test.local", "name": "Test Admin",
            "role": role, "organization_id": org}
    return {"Authorization": f"Bearer {TEST_PREFIX}{json.dumps(user)}"}


# Rate limits (A5) are process-wide; start every test with empty windows so
# the suite's own requests never count against each other.
import pytest
from api.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    limiter.reset()
    yield
    limiter.reset()
