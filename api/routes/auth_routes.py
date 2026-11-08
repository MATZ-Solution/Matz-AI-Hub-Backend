"""api/routes/auth_routes.py — credential check for the web app's sign-in (A3).

Called only by the Next.js server (Auth.js Credentials provider) with a
60-second "login" token signed with AUTH_SECRET; see api/auth.py.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from agents.langgraph_agent.utils.utils import logger
from api.auth import verify_password
from api.errors import ApiError, ErrorCode
from api.helpers import user_store

router = APIRouter(tags=["auth"])

WRONG_CREDENTIALS = "Email or password is incorrect"


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    organization_id: str


@router.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest):
    user = user_store.get_user_by_email(request.email)
    # Always run the hash check, even for an unknown email, so timing does not
    # reveal which emails have accounts. One message for every failure.
    ok = verify_password(user["password_hash"] if user else None, request.password)
    if not ok or not user or not user.get("is_active", True):
        logger.info("Auth → failed login for %s", request.email.strip().lower()[:200])
        raise ApiError(ErrorCode.unauthorized, WRONG_CREDENTIALS)

    try:
        user_store.record_login(user["id"])
    except Exception as e:  # bookkeeping only; never block a valid sign-in
        logger.warning("Auth → could not record login: %s", e)

    logger.info("Auth → login %s (%s)", user["email"], user["role"])
    return LoginResponse(
        id=str(user["id"]), email=user["email"], name=user.get("name") or "",
        role=user["role"], organization_id=user["organization_id"],
    )
