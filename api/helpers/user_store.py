"""
api/helpers/user_store.py
---------------------------
Reads and writes the `users` table (supabase/migrations/002_users.sql).
Errors propagate: callers decide what the user sees.
"""

from datetime import datetime, timezone

from agents.langgraph_agent.utils.utils import get_supabase, _exec

PUBLIC_FIELDS = "id,email,name,role,organization_id,is_active"


def get_user_by_email(email: str) -> dict | None:
    """The account for this email (case-insensitive), including password_hash.
    Emails are always stored lowercased (create_user), so an exact match on
    the lowercased input is a case-insensitive lookup."""
    rows = _exec(get_supabase().table("users").select("*")
                 .eq("email", email.strip().lower()).limit(1)).data
    return rows[0] if rows else None


def create_user(*, email: str, name: str, password_hash: str, role: str, organization_id: str) -> dict:
    row = {
        "email": email.strip().lower(),
        "name": name,
        "password_hash": password_hash,
        "role": role,
        "organization_id": organization_id,
    }
    return _exec(get_supabase().table("users").insert(row)).data[0]


def record_login(user_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    _exec(get_supabase().table("users")
          .update({"last_login_at": now, "failed_logins": 0, "updated_at": now})
          .eq("id", user_id))


def get_user(user_id: str, organization_id: str) -> dict | None:
    """The account with this id in this organization (never across orgs)."""
    rows = _exec(get_supabase().table("users").select("*")
                 .eq("id", user_id).eq("organization_id", organization_id).limit(1)).data
    return rows[0] if rows else None


def update_user(user_id: str, organization_id: str, values: dict) -> dict | None:
    values = {**values, "updated_at": datetime.now(timezone.utc).isoformat()}
    rows = _exec(get_supabase().table("users").update(values)
                 .eq("id", user_id).eq("organization_id", organization_id)).data
    return rows[0] if rows else None


# ── Sessions (supabase/migrations/003_auth_sessions.sql) ─────────────────────

def create_session(user_id: str, organization_id: str, token_hash: str, ttl_seconds: int) -> datetime:
    """Stores a new session; returns when it expires."""
    from datetime import timedelta
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    _exec(get_supabase().table("auth_sessions").insert({
        "token_hash": token_hash, "user_id": user_id,
        "organization_id": organization_id, "expires_at": expires_at.isoformat(),
    }))
    return expires_at


def get_session_user(token_hash: str) -> tuple[dict, float] | None:
    """(active user row, session expiry as epoch seconds) for a live session,
    or None if the session is unknown, revoked or expired, or the user is
    deactivated or no longer in the session's organization."""
    rows = _exec(get_supabase().table("auth_sessions").select("*")
                 .eq("token_hash", token_hash).limit(1)).data
    if not rows or rows[0].get("revoked_at"):
        return None
    session = rows[0]
    expires = datetime.fromisoformat(str(session["expires_at"]).replace("Z", "+00:00"))
    if expires <= datetime.now(timezone.utc):
        return None
    user = get_user(str(session["user_id"]), session["organization_id"])
    if not user or not user.get("is_active", True):
        return None
    return user, expires.timestamp()


def revoke_session(token_hash: str) -> None:
    _exec(get_supabase().table("auth_sessions")
          .update({"revoked_at": datetime.now(timezone.utc).isoformat()})
          .eq("token_hash", token_hash))


def revoke_other_sessions(user_id: str, keep_token_hash: str | None) -> None:
    """Signs the user out everywhere else (after a password change)."""
    rows = _exec(get_supabase().table("auth_sessions").select("token_hash,revoked_at")
                 .eq("user_id", user_id)).data
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if row["token_hash"] != keep_token_hash and not row.get("revoked_at"):
            _exec(get_supabase().table("auth_sessions").update({"revoked_at": now})
                  .eq("token_hash", row["token_hash"]))
