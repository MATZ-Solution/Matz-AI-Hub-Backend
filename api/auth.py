"""
api/auth.py
------------
Authentication for the API (task A3).

The web app (Auth.js v5) signs a session JWT with the shared AUTH_SECRET
(HS256) and the Next.js server forwards it as `Authorization: Bearer <jwt>`.
This module verifies that token on every route except /health. The org and
role used by the API come from the verified token only.

Token purposes:
  - "session": a signed-in user's session (8 h), sent on every API call.
  - "login":   a 60-second token the web server mints to call POST /auth/login,
               so even the login endpoint is not open to anonymous callers.
"""

import os
import time
from dataclasses import dataclass

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Request

from agents.langgraph_agent.utils.utils import logger
from api.errors import ErrorCode, error_response

JWT_ALGORITHM = "HS256"
JWT_LEEWAY_SECONDS = 30

# Paths that never need a token. Everything else does (task A3).
PUBLIC_PATHS = {"/health"}
LOGIN_PATH = "/auth/login"

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


# ── Tokens ───────────────────────────────────────────────────────────────────

def _secret() -> str:
    secret = os.environ.get("AUTH_SECRET", "")
    if not secret:
        raise RuntimeError("AUTH_SECRET is not set")
    return secret


def decode_token(token: str, purpose: str) -> dict | None:
    """Claims of a valid, unexpired token with this purpose; None otherwise."""
    try:
        claims = jwt.decode(
            token, _secret(), algorithms=[JWT_ALGORITHM], leeway=JWT_LEEWAY_SECONDS,
            options={"require": ["exp", "iat"]},
        )
    except jwt.PyJWTError:
        return None
    return claims if claims.get("purpose") == purpose else None


def make_token(claims: dict, purpose: str, ttl_seconds: int) -> str:
    """Used by tests and scripts; the web app signs its own tokens."""
    now = int(time.time())
    payload = {**claims, "purpose": purpose, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def user_from_claims(claims: dict) -> CurrentUser | None:
    try:
        return CurrentUser(
            id=str(claims["sub"]),
            email=str(claims["email"]),
            name=str(claims.get("name") or ""),
            role=str(claims["role"]),
            organization_id=str(claims["org_id"]),
        )
    except KeyError:
        return None


def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


# ── Middleware + dependency ──────────────────────────────────────────────────

def install_auth(app, docs_paths: set[str]) -> None:
    """Rejects every request without a valid token, except PUBLIC_PATHS (and
    the API docs when they are enabled). Must be installed before the error
    handling middleware so 401s still carry a request id."""
    open_paths = PUBLIC_PATHS | docs_paths

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path in open_paths:
            return await call_next(request)

        token = _bearer(request)
        if path == LOGIN_PATH:
            if not token or decode_token(token, "login") is None:
                return error_response(ErrorCode.unauthorized)
            return await call_next(request)

        claims = decode_token(token, "session") if token else None
        user = user_from_claims(claims) if claims else None
        if user is None:
            return error_response(ErrorCode.unauthorized)
        request.state.user = user
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
