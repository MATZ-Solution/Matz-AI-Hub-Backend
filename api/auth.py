"""
api/auth.py
------------
Authentication for the API.

Sign-in checks the email and password against the `users` table (argon2id
hashes). A successful sign-in creates a row in `auth_sessions` and returns a
random 256-bit token; the web app keeps it in an httpOnly cookie and sends
it as `Authorization: Bearer <token>` on every call. Only the token's SHA-256
hash is stored, and no shared secret is needed between the web app and the
API (see supabase/migrations/003_auth_sessions.sql).

Every route except /health and /auth/login requires a valid, unexpired,
unrevoked session of an active user. The organization and role used by the
API come from that session's user only.
"""

import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Request

from agents.langgraph_agent.utils.utils import logger
from api.errors import ErrorCode, error_response

SESSION_SECONDS = 8 * 60 * 60

# Paths that need no session. Everything else does.
PUBLIC_PATHS = {"/health", "/auth/login"}

# A verified session is reused for this long before the database is asked
# again, so a revoked session (sign-out, deactivation) stops working within
# this window at most. Expiry is always checked exactly.
SESSION_CACHE_SECONDS = 30

# Defaults of argon2-cffi: argon2id, 64 MiB memory, time cost 3.
_hasher = PasswordHasher()
# Verified against when the email is unknown, so a wrong email and a wrong
# password take the same time and cannot be told apart.
_DUMMY_HASH = _hasher.hash("timing-equaliser-not-a-real-password")


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str
    name: str
    role: str
    organization_id: str


# ── Passwords ────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# ── Session tokens ───────────────────────────────────────────────────────────

def new_session_token() -> str:
    return secrets.token_urlsafe(32)  # 256 bits


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# token hash → (cached_at, CurrentUser, expires_at epoch)
_session_cache: dict[str, tuple[float, CurrentUser, float]] = {}
_cache_lock = threading.Lock()


def forget_session(token: str) -> None:
    with _cache_lock:
        _session_cache.pop(hash_token(token), None)


def lookup_session(token: str) -> CurrentUser | None:
    """The active user behind this session token, or None."""
    from api.helpers import user_store

    token_hash = hash_token(token)
    now = time.time()
    with _cache_lock:
        cached = _session_cache.get(token_hash)
    if cached and now - cached[0] < SESSION_CACHE_SECONDS and now < cached[2]:
        return cached[1]

    found = user_store.get_session_user(token_hash)
    if not found:
        with _cache_lock:
            _session_cache.pop(token_hash, None)
        return None
    row, expires_at = found
    user = CurrentUser(id=str(row["id"]), email=row["email"], name=row.get("name") or "",
                       role=row["role"], organization_id=row["organization_id"])
    with _cache_lock:
        _session_cache[token_hash] = (now, user, expires_at)
    return user


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


# ── Middleware + dependency ──────────────────────────────────────────────────

def install_auth(app, docs_paths: set[str]) -> None:
    """Rejects every request without a valid session, except PUBLIC_PATHS (and
    the API docs when they are enabled). Must be installed before the error
    handling middleware so 401s still carry a request id."""
    open_paths = PUBLIC_PATHS | docs_paths

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in open_paths:
            return await call_next(request)

        token = bearer_token(request)
        try:
            user = lookup_session(token) if token else None
        except Exception as e:  # session store unreachable: fail closed
            logger.error("Auth → session lookup failed: %s", e)
            return error_response(ErrorCode.internal)
        if user is None:
            return error_response(ErrorCode.unauthorized)
        request.state.user = user
        request.state.session_token = token
        return await call_next(request)


def get_current_user(request: Request) -> CurrentUser:
    """FastAPI dependency: the verified user for this request."""
    return request.state.user


# ── First admin ──────────────────────────────────────────────────────────────

def seed_admin() -> None:
    """Creates the first admin from ADMIN_EMAIL / ADMIN_PASSWORD if no account
    with that email exists. Never changes an existing account's password."""
    from api.config.settings import DEFAULT_ORGANIZATION_ID
    from api.helpers import user_store

    email = (os.environ.get("ADMIN_EMAIL") or "").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD") or ""
    if not email or not password:
        logger.warning("Auth → ADMIN_EMAIL/ADMIN_PASSWORD not set; no admin seeded")
        return
    try:
        if user_store.get_user_by_email(email):
            return
        user_store.create_user(
            email=email, name="Admin", password_hash=hash_password(password),
            role="admin", organization_id=DEFAULT_ORGANIZATION_ID,
        )
        logger.info("Auth → seeded first admin %s", email)
    except Exception as e:
        logger.error("Auth → could not seed admin (is migration 002_users.sql applied?): %s", e)
