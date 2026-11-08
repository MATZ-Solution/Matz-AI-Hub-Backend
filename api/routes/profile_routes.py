"""api/routes/profile_routes.py — the signed-in user's own profile (task B4).

GET /me and PATCH /me read and rename the current user; POST /me/password
changes their password after checking the current one. The user is always
the one in the verified token, so nobody can edit someone else's profile.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from agents.langgraph_agent.utils.utils import logger
from api.auth import CurrentUser, get_current_user, hash_password, verify_password
from api.errors import ApiError, ErrorCode
from api.helpers import user_store

router = APIRouter(tags=["profile"])

PASSWORD_MIN, PASSWORD_MAX = 10, 128


class Profile(BaseModel):
    id: str
    email: str
    name: str
    role: str
    organization_id: str


class ProfileUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def _profile(row: dict) -> Profile:
    return Profile(id=str(row["id"]), email=row["email"], name=row.get("name") or "",
                   role=row["role"], organization_id=row["organization_id"])


def _load(user: CurrentUser) -> dict:
    row = user_store.get_user(user.id, user.organization_id)
    if not row or not row.get("is_active", True):
        raise ApiError(ErrorCode.unauthorized)
    return row


@router.get("/me", response_model=Profile)
def get_me(user: CurrentUser = Depends(get_current_user)):
    return _profile(_load(user))


@router.patch("/me", response_model=Profile)
def update_me(request: ProfileUpdate, user: CurrentUser = Depends(get_current_user)):
    name = request.name.strip()
    if not name:
        raise ApiError(ErrorCode.invalid_request, "Enter your name.")
    _load(user)
    row = user_store.update_user(user.id, user.organization_id, {"name": name})
    logger.info("Profile → name updated for %s", user.id)
    return _profile(row)


@router.post("/me/password")
def change_password(request: PasswordChange, user: CurrentUser = Depends(get_current_user)):
    row = _load(user)
    if not verify_password(row["password_hash"], request.current_password):
        raise ApiError(ErrorCode.invalid_request, "Your current password is incorrect.")
    new = request.new_password
    if not PASSWORD_MIN <= len(new) <= PASSWORD_MAX:
        raise ApiError(ErrorCode.invalid_request,
                       f"Use a password between {PASSWORD_MIN} and {PASSWORD_MAX} characters.")
    if new == request.current_password:
        raise ApiError(ErrorCode.invalid_request, "Choose a password different from your current one.")
    user_store.update_user(user.id, user.organization_id, {"password_hash": hash_password(new)})
    logger.info("Profile → password changed for %s", user.id)
    return {"success": True}
