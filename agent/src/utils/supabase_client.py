"""
src/utils/supabase_client.py
-----------------------------
Supabase client for chat history and collections persistence.

Tables:
  sessions    — one per conversation
  messages    — user and assistant messages per session
  collections — company knowledge collections
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from agent.src.utils.logger import logger

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

_client = None


def get_supabase():
    """Lazy singleton Supabase client."""
    global _client
    if _client is None:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def check_supabase_connection() -> bool:
    """Pings Supabase on startup so connection issues surface immediately."""
    try:
        client = get_supabase()
        client.table("sessions").select("id").limit(1).execute()
        logger.info("Supabase → connected")
        return True
    except Exception as e:
        logger.error("Supabase → connection failed: %s", e)
        return False


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(organization_id: str = "matz-demo-org") -> str:
    """Creates a new chat session. Returns the session ID."""
    try:
        client = get_supabase()
        result = client.table("sessions").insert({
            "organization_id": organization_id
        }).execute()
        session_id = result.data[0]["id"]
        logger.info("Supabase → session created: %s", session_id)
        return session_id
    except Exception as e:
        logger.error("Supabase → create_session failed: %s", e)
        return None


def get_sessions(organization_id: str = "matz-demo-org") -> list:
    """Returns all sessions for an organization, most recent first."""
    try:
        client = get_supabase()
        result = client.table("sessions") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .order("created_at", desc=True) \
            .execute()
        return result.data
    except Exception as e:
        logger.error("Supabase → get_sessions failed: %s", e)
        return []


def delete_session(session_id: str) -> bool:
    """
    Deletes a session and all its messages (cascade delete).
    Returns True if successful.
    """
    try:
        client = get_supabase()
        client.table("sessions").delete().eq("id", session_id).execute()
        logger.info("Supabase → session deleted: %s", session_id)
        return True
    except Exception as e:
        logger.error("Supabase → delete_session failed: %s", e)
        return False


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(
    session_id: str, role: str, content: str, citations: list = None,
    is_grounded: bool = None, response_time_ms: int = None,
) -> str:
    """
    Saves a single message to Supabase. Returns the message ID.

    is_grounded and response_time_ms are optional — only meaningful for
    assistant messages, and only populated going forward (older rows have
    NULL here, which analytics queries treat as "not tracked yet").
    """
    try:
        client = get_supabase()
        payload = {
            "session_id": session_id,
            "role":       role,
            "content":    content,
            "citations":  citations or [],
        }
        if is_grounded is not None:
            payload["is_grounded"] = is_grounded
        if response_time_ms is not None:
            payload["response_time_ms"] = response_time_ms

        result = client.table("messages").insert(payload).execute()
        message_id = result.data[0]["id"]
        logger.info("Supabase → message saved: %s (%s)", message_id, role)
        return message_id
    except Exception as e:
        logger.error("Supabase → save_message failed: %s", e)
        return None


def get_messages(session_id: str) -> list:
    """Returns all messages for a session, oldest first."""
    try:
        client = get_supabase()
        result = client.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()
        return result.data
    except Exception as e:
        logger.error("Supabase → get_messages failed: %s", e)
        return []


# ── Collections ───────────────────────────────────────────────────────────────

def get_collections_from_db(organization_id: str = "matz-demo-org") -> list:
    """Returns all collections from Supabase."""
    try:
        client = get_supabase()
        result = client.table("collections") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .order("created_at", desc=False) \
            .execute()
        return result.data
    except Exception as e:
        logger.error("Supabase → get_collections failed: %s", e)
        return []


def create_collection_in_db(
    name: str,
    description: str = "",
    icon: str = "folder",
    organization_id: str = "matz-demo-org"
) -> dict:
    """Creates a new collection in Supabase."""
    try:
        client = get_supabase()
        result = client.table("collections").insert({
            "name":            name,
            "description":     description,
            "icon":            icon,
            "organization_id": organization_id,
        }).execute()
        logger.info("Supabase → collection created: %s", name)
        return result.data[0]
    except Exception as e:
        logger.error("Supabase → create_collection failed: %s", e)
        return None


def delete_collection_from_db(collection_id: str) -> bool:
    """Deletes a collection from Supabase."""
    try:
        client = get_supabase()
        client.table("collections").delete().eq("id", collection_id).execute()
        logger.info("Supabase → collection deleted: %s", collection_id)
        return True
    except Exception as e:
        logger.error("Supabase → delete_collection failed: %s", e)
        return False

# ── Workspace settings ────────────────────────────────────────────────────────

def get_workspace_settings(organization_id: str = "matz-demo-org") -> dict:
    """Returns workspace settings from Supabase, or sane defaults if none saved yet."""
    try:
        client = get_supabase()
        result = client.table("workspace_settings") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .limit(1).execute()
        if result.data:
            return result.data[0]
        return {"organization_id": organization_id, "name": "", "url": ""}
    except Exception as e:
        logger.error("Supabase → get_workspace_settings failed: %s", e)
        return {"organization_id": organization_id, "name": "", "url": ""}


def upsert_workspace_settings(organization_id: str, name: str, url: str) -> dict:
    """Creates or updates workspace settings for an organization."""
    try:
        client = get_supabase()
        result = client.table("workspace_settings").upsert({
            "organization_id": organization_id,
            "name": name,
            "url": url,
        }).execute()
        logger.info("Supabase → workspace settings saved: %s", organization_id)
        return result.data[0]
    except Exception as e:
        logger.error("Supabase → upsert_workspace_settings failed: %s", e)
        return None


# ── Assistant config ─────────────────────────────────────────────────────────

def get_assistant_config(organization_id: str = "matz-demo-org") -> dict:
    """Returns assistant config from Supabase, or sane defaults if none saved yet."""
    try:
        client = get_supabase()
        result = client.table("assistant_config") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .limit(1).execute()
        if result.data:
            return result.data[0]
        return {
            "organization_id": organization_id,
            "name": "MATZ Assistant",
            "personality": "Helpful and precise",
            "instructions": "",
        }
    except Exception as e:
        logger.error("Supabase → get_assistant_config failed: %s", e)
        return {
            "organization_id": organization_id,
            "name": "MATZ Assistant",
            "personality": "Helpful and precise",
            "instructions": "",
        }


def upsert_assistant_config(organization_id: str, name: str, personality: str, instructions: str) -> dict:
    """Creates or updates the assistant configuration for an organization."""
    try:
        client = get_supabase()
        result = client.table("assistant_config").upsert({
            "organization_id": organization_id,
            "name": name,
            "personality": personality,
            "instructions": instructions,
        }).execute()
        logger.info("Supabase → assistant config saved: %s", organization_id)
        return result.data[0]
    except Exception as e:
        logger.error("Supabase → upsert_assistant_config failed: %s", e)
        return None


# ── Members ───────────────────────────────────────────────────────────────────

def get_members(organization_id: str = "matz-demo-org") -> list:
    """Returns all members for an organization, oldest first."""
    try:
        client = get_supabase()
        result = client.table("members") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .order("created_at", desc=False) \
            .execute()
        return result.data
    except Exception as e:
        logger.error("Supabase → get_members failed: %s", e)
        return []


def create_member(organization_id: str, name: str, email: str, role: str = "Viewer") -> dict:
    """Adds a member to an organization."""
    try:
        client = get_supabase()
        result = client.table("members").insert({
            "organization_id": organization_id,
            "name": name,
            "email": email,
            "role": role,
        }).execute()
        logger.info("Supabase → member added: %s", email)
        return result.data[0]
    except Exception as e:
        logger.error("Supabase → create_member failed: %s", e)
        return None


def delete_member(member_id: str) -> bool:
    """Removes a member."""
    try:
        client = get_supabase()
        client.table("members").delete().eq("id", member_id).execute()
        logger.info("Supabase → member deleted: %s", member_id)
        return True
    except Exception as e:
        logger.error("Supabase → delete_member failed: %s", e)
        return False


# ── Usage ─────────────────────────────────────────────────────────────────────

def count_user_questions(organization_id: str = "matz-demo-org") -> int:
    """Counts user messages across all sessions for an organization."""
    try:
        client = get_supabase()
        session_ids = [
            s["id"] for s in
            client.table("sessions").select("id").eq("organization_id", organization_id).execute().data
        ]
        if not session_ids:
            return 0
        result = client.table("messages") \
            .select("id", count="exact") \
            .in_("session_id", session_ids) \
            .eq("role", "user") \
            .execute()
        return result.count or 0
    except Exception as e:
        logger.error("Supabase → count_user_questions failed: %s", e)
        return 0


# ── Analytics ─────────────────────────────────────────────────────────────────

def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _pct_change(previous: int, current: int):
    """
    None means 'no baseline to compare against' — the frontend renders that
    as '—'.

    A zero previous period is NOT "+100% growth": there is simply nothing to
    compare against. Returning a real-looking percentage there implies a trend
    that doesn't exist, so return None until there's an actual prior number.
    """
    if previous == 0:
        return None
    return round((current - previous) / previous * 100, 1)


def _empty_analytics(organization_id: str, days: int) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "questions_asked": 0,
        "questions_asked_change_pct": None,
        "successful_answers": 0,
        "success_rate_pct": None,
        "avg_response_time_ms": None,
        "avg_response_time_change_ms": None,
        "questions_over_time": [
            {"date": (now - timedelta(days=days - 1 - i)).date().isoformat(), "count": 0}
            for i in range(days)
        ],
        "popular_collections": [],
        "organization_id": organization_id,
    }


def get_analytics_summary(organization_id: str = "matz-demo-org", days: int = 14) -> dict:
    """
    Computes real usage/performance analytics from Supabase message and
    session history for the last `days` days, compared against the `days`
    before that for trend percentages.

    Note: success_rate_pct and avg_response_time_ms only reflect messages
    sent after is_grounded/response_time_ms tracking was added — older rows
    have those columns as NULL and are excluded, not counted as failures.
    """
    try:
        client = get_supabase()

        now = datetime.now(timezone.utc)
        current_start = now - timedelta(days=days)
        previous_start = now - timedelta(days=days * 2)

        sessions = client.table("sessions") \
            .select("id") \
            .eq("organization_id", organization_id) \
            .gte("created_at", previous_start.isoformat()) \
            .execute().data
        session_ids = [s["id"] for s in sessions]

        if not session_ids:
            return _empty_analytics(organization_id, days)

        messages = client.table("messages") \
            .select("session_id, role, citations, is_grounded, response_time_ms, created_at") \
            .in_("session_id", session_ids) \
            .gte("created_at", previous_start.isoformat()) \
            .order("created_at", desc=False) \
            .execute().data

        def in_range(msg, start, end):
            ts = _parse_ts(msg["created_at"])
            return start <= ts < end

        current_msgs  = [m for m in messages if in_range(m, current_start, now)]
        previous_msgs = [m for m in messages if in_range(m, previous_start, current_start)]

        # ── Questions asked ──────────────────────────────────────────────────
        current_questions  = [m for m in current_msgs  if m["role"] == "user"]
        previous_questions = [m for m in previous_msgs if m["role"] == "user"]
        questions_asked = len(current_questions)
        questions_asked_change_pct = _pct_change(len(previous_questions), questions_asked)

        # ── Questions over time (daily buckets, zero-filled) ─────────────────
        buckets = {
            (current_start + timedelta(days=i)).date().isoformat(): 0
            for i in range(days)
        }
        for m in current_questions:
            day = _parse_ts(m["created_at"]).date().isoformat()
            if day in buckets:
                buckets[day] += 1
        questions_over_time = [{"date": d, "count": c} for d, c in sorted(buckets.items())]

        # ── Successful answers (grounded rate) ────────────────────────────────
        current_checked = [
            m for m in current_msgs
            if m["role"] == "assistant" and m.get("is_grounded") is not None
        ]
        successful_answers = sum(1 for m in current_checked if m["is_grounded"])
        success_rate_pct = (
            round(successful_answers / len(current_checked) * 100, 1) if current_checked else None
        )

        # ── Avg response time ───────────────────────────────────────────────
        current_timed = [
            m["response_time_ms"] for m in current_msgs
            if m["role"] == "assistant" and m.get("response_time_ms") is not None
        ]
        previous_timed = [
            m["response_time_ms"] for m in previous_msgs
            if m["role"] == "assistant" and m.get("response_time_ms") is not None
        ]
        avg_response_time_ms = round(sum(current_timed) / len(current_timed)) if current_timed else None
        prev_avg_response_time_ms = (
            round(sum(previous_timed) / len(previous_timed)) if previous_timed else None
        )
        avg_response_time_change_ms = (
            avg_response_time_ms - prev_avg_response_time_ms
            if avg_response_time_ms is not None and prev_avg_response_time_ms is not None
            else None
        )

        # ── Popular collections (from citations on current-period answers) ──
        collection_counts: dict = {}
        for m in current_msgs:
            if m["role"] != "assistant":
                continue
            for citation in (m.get("citations") or []):
                name = citation.get("collection_name") or "Uncategorized"
                collection_counts[name] = collection_counts.get(name, 0) + 1
        total_citations = sum(collection_counts.values())
        popular_collections = []
        if total_citations:
            top = sorted(collection_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
            popular_collections = [
                {"name": name, "percent": round(count / total_citations * 100, 1)}
                for name, count in top
            ]

        return {
            "questions_asked": questions_asked,
            "questions_asked_change_pct": questions_asked_change_pct,
            "successful_answers": successful_answers,
            "success_rate_pct": success_rate_pct,
            "avg_response_time_ms": avg_response_time_ms,
            "avg_response_time_change_ms": avg_response_time_change_ms,
            "questions_over_time": questions_over_time,
            "popular_collections": popular_collections,
            "organization_id": organization_id,
        }
    except Exception as e:
        logger.error("Supabase → get_analytics_summary failed: %s", e)
        return _empty_analytics(organization_id, days)