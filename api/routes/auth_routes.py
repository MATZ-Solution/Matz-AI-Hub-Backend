"""api/routes/auth_routes.py — sign-in and sign-out.

POST /auth/login checks the email and password against the users table and,
on success, starts a session (api/auth.py). POST /auth/logout ends the
current session. The web app keeps the token in an httpOnly cookie.
"""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from agents.langgraph_agent.utils.utils import logger
from api.auth import (
    SESSION_SECONDS, CurrentUser, forget_session, get_current_user, hash_token,
    new_session_token, verify_password,
)
from api.errors import ApiError, ErrorCode
from api.helpers import user_store

router = APIRouter(tags=["auth"])

WRONG_CREDENTIALS = "Email or password is incorrect"


class LoginRequest(BaseModel):
    email: str
    password: str


class SessionUser(BaseModel):
    id: str
    email: str
    name: str
    role: str
    organization_id: str


class LoginResponse(BaseModel):
    token: str
    expires_at: str
    user: SessionUser


@router.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest):
    user = user_store.get_user_by_email(request.email)
    # Always run the hash check, even for an unknown email, so timing does not
    # reveal which emails have accounts. One message for every failure.
    ok = verify_password(user["password_hash"] if user else None, request.password)
    if not ok or not user or not user.get("is_active", True):
        logger.info("Auth → failed login for %s", request.email.strip().lower()[:200])
        raise ApiError(ErrorCode.unauthorized, WRONG_CREDENTIALS)

    token = new_session_token()
    expires_at = user_store.create_session(
        str(user["id"]), user["organization_id"], hash_token(token), SESSION_SECONDS,
    )
    try:
        user_store.record_login(user["id"])
    except Exception as e:  # bookkeeping only; never block a valid sign-in
        logger.warning("Auth → could not record login: %s", e)

    logger.info("Auth → login %s (%s)", user["email"], user["role"])
    return LoginResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        user=SessionUser(id=str(user["id"]), email=user["email"], name=user.get("name") or "",
                         role=user["role"], organization_id=user["organization_id"]),
    )


@router.post("/auth/logout")
def logout(request: Request, user: CurrentUser = Depends(get_current_user)):
    token = request.state.session_token
    user_store.revoke_session(hash_token(token))
    forget_session(token)
    logger.info("Auth → logout %s", user.email)
    return {"success": True}
