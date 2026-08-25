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

def save_message(session_id: str, role: str, content: str, citations: list = None) -> str:
    """Saves a single message to Supabase. Returns the message ID."""
    try:
        client = get_supabase()
        result = client.table("messages").insert({
            "session_id": session_id,
            "role":       role,
            "content":    content,
            "citations":  citations or [],
        }).execute()
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