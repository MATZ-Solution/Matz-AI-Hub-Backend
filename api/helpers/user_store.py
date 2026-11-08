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
