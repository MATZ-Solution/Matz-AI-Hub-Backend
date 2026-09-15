"""Shared utilities: logging, config, cache, spell-correction, summary, Supabase client."""

# ==========================================================================
# logger
# ==========================================================================

"""
src/utils/logger.py
--------------------
Centralised logger for the entire project.
Import `logger` anywhere instead of using bare print() statements.

Usage:
    logger.info("Graph compiled successfully")
    logger.error("LLM call failed: %s", str(e))
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "matz_agent") -> logging.Logger:
    """Return a configured logger. Call once per module."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


# Default logger — import this directly in most cases
logger = get_logger()


# ==========================================================================
# config
# ==========================================================================

"""
src/utils/config.py
--------------------
Single place to read and validate all environment variables.
Import `cfg` anywhere instead of sprinkling os.environ.get() calls
across the codebase.

Usage:
    print(cfg.qdrant_url)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Qdrant
    qdrant_url: str     = os.environ.get("QDRANT_URL", "")
    qdrant_api_key: str = os.environ.get("QDRANT_API_KEY", "")

    # Groq
    groq_api_key: str   = os.environ.get("GROQ_API_KEY", "")

    # Agent settings
    collection_name: str    = "matz_chunks"
    embed_model: str        = "all-MiniLM-L6-v2"
    llm_model: str          = "openai/gpt-oss-120b"
    top_k: int              = 3
    min_score: float        = 0.2
    organization_id: str    = "matz-demo-org"
    max_history_messages: int = 20


# Singleton — import this directly
cfg = Config()


# ==========================================================================
# spell_correct
# ==========================================================================

"""
src/utils/spell_correct.py
---------------------------
Spell correction for English queries only.
Skips correction if Urdu words are detected.
"""
from textblob import TextBlob

URDU_MARKERS = [
    "kya", "hai", "kiya", "batao", "mujhe", "mujhey",
    "karo", "hain", "ka", "ki", "ke", "ho", "aur",
    "nahi", "kuch", "bhi", "se", "mein", "ko",
]

def correct_query(text: str) -> str:
    """Correct spelling only for English queries. Skip if Roman Urdu detected."""
    text_lower = text.lower()

    # If any Urdu marker found → skip correction entirely
    if any(marker in text_lower.split() for marker in URDU_MARKERS):
        return text

    try:
        corrected = str(TextBlob(text).correct())
        return corrected
    except Exception:
        return text


# ==========================================================================
# summary
# ==========================================================================

"""
src/utils/summarizer.py
------------------------
Chat history summarizer.

What it does:
- When conversation history reaches SUMMARIZE_AFTER messages,
  compresses the oldest MESSAGES_TO_SUMMARIZE messages into
  one short summary paragraph.
- Keeps the most recent MESSAGES_TO_KEEP messages as-is for context.
- Uses llama-3.1-8b-instant (fast, cheap — summarization doesn't need big model)

Example:
  History: 10 messages
  → Summarize oldest 6 → 1 summary paragraph (~50 tokens)
  → Keep newest 4 as-is
  → New history: 1 summary + 4 recent = 5 messages
  → Token savings: ~70%

Why this matters:
  Without summarizer: 20 messages × 200 tokens = 4000 tokens per request
  With summarizer:    1 summary + 4 messages  = ~900 tokens per request
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ── Config ────────────────────────────────────────────────────────────────────
SUMMARIZE_AFTER      = 10  # trigger summarization when history reaches this many messages
MESSAGES_TO_SUMMARIZE = 6  # how many old messages to compress
MESSAGES_TO_KEEP      = 4  # how many recent messages to keep as-is

# ── Summarizer prompt ─────────────────────────────────────────────────────────
SUMMARIZER_PROMPT = (
    "You are a conversation summarizer for a company AI knowledge assistant.\n\n"
    "Given a list of conversation messages, write a concise summary in 2-3 sentences "
    "that captures:\n"
    "1. What topics the user asked about\n"
    "2. What key information was provided (policies, rules, numbers)\n"
    "3. Any important context for continuing the conversation\n\n"
    "Rules:\n"
    "- Be factual — only include what was actually discussed\n"
    "- Keep it under 100 words\n"
    "- Write in third person: 'The user asked about...'\n"
    "- Do NOT include greetings or small talk\n"
    "- Focus on company policy information that was shared\n"
    "- Respond with ONLY the summary paragraph, nothing else"
)


def maybe_summarize(chat_history: list) -> list:
    """
    Check if history is long enough to summarize.
    If yes — compress old messages and return shortened history.
    If no — return history unchanged.

    Called at the START of generate_answer before building messages_for_llm.
    """
    if len(chat_history) < SUMMARIZE_AFTER:
        return chat_history

    logger.info(
        "Summarizer → history has %d messages, summarizing oldest %d",
        len(chat_history), MESSAGES_TO_SUMMARIZE
    )

    # Split history into old (to summarize) and recent (to keep)
    old_messages    = chat_history[:MESSAGES_TO_SUMMARIZE]
    recent_messages = chat_history[MESSAGES_TO_SUMMARIZE:]

    # Summarize the old messages
    summary_text = _summarize_messages(old_messages)

    if summary_text:
        # Replace old messages with one summary message
        summary_message = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY — earlier in this session]: {summary_text}"
        }
        new_history = [summary_message] + recent_messages
        logger.info(
            "Summarizer → compressed %d messages into 1 summary (%d messages total)",
            MESSAGES_TO_SUMMARIZE, len(new_history)
        )
        return new_history

    # If summarization failed — return original history
    logger.warning("Summarizer → failed, returning original history")
    return chat_history


def _summarize_messages(messages: list) -> str:
    """
    Ask the LLM to summarize a list of messages.
    Uses llama-3.1-8b-instant — fast and cheap for simple summarization.
    """
    try:
        from agents.langgraph_agent.LLMs.llm import call_llm

        # Format messages into readable text for summarization
        conversation_text = ""
        for msg in messages:
            role    = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            # Skip system messages and very short messages
            if role == "System" or len(content) < 10:
                continue
            conversation_text += f"{role}: {content}\n\n"

        if not conversation_text.strip():
            return ""

        summary = call_llm(
            system_prompt=SUMMARIZER_PROMPT,
            messages=[{"role": "user", "content": f"Conversation to summarize:\n\n{conversation_text}"}],
            model=os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b"),   # fast model — summarization is simple
            temperature=0.0,
            max_tokens=150,
        )
        return summary.strip()

    except Exception as e:
        logger.warning("Summarizer LLM failed (%s)", e)
        return ""


# ==========================================================================
# supabase_client
# ==========================================================================

"""
src/utils/supabase_client.py
-----------------------------
Supabase client for chat history and collections persistence.

Tables:
  sessions           — one per conversation
  messages           — user and assistant messages per session
  collections        — company knowledge collections
  workspace_settings — workspace identity (name)
  assistant_config   — configurable assistant name/personality/instructions
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# ── Client ────────────────────────────────────────────────────────────────────
#
# One client PER THREAD, not one shared globally.
#
# The API handlers are plain `def`, so FastAPI runs them in a threadpool and
# several can hit Supabase at the same moment. A single shared client means
# several threads sharing one httpx connection pool, which on Windows surfaces
# as "[WinError 10035] A non-blocking socket operation could not be completed
# immediately" and spurious read timeouts. threading.local() gives each worker
# thread its own client and its own connections, so requests never interleave
# on the same socket.
#
# FastAPI's threadpool is bounded (40 threads by default), so this creates at
# most that many clients for the life of the process.

import threading

_local = threading.local()

# Measured behaviour of this connection (see measure_supabase.py): successful
# calls return in ~0.18s median, 0.40s worst case. Failures do not come back at
# all — roughly 13% of connections are silently dropped somewhere in the path.
#
# Because the distribution is bimodal (fast or never), waiting longer never
# helps: a request that will never arrive does not arrive at 30s either. The
# correct response is to give up quickly and try again on a fresh connection.
#
# 5s is >10x the observed worst case, so a timeout means "dropped", not "slow".
# With 3 attempts a 13% per-call drop rate becomes ~0.2% overall.
SUPABASE_TIMEOUT_SECONDS = 5
SUPABASE_MAX_ATTEMPTS = 3


def get_supabase():
    """Returns this thread's Supabase client, creating it on first use."""
    client = getattr(_local, "client", None)
    if client is not None:
        return client

    from supabase import create_client
    try:
        from supabase import ClientOptions
        client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options=ClientOptions(
                postgrest_client_timeout=SUPABASE_TIMEOUT_SECONDS,
                storage_client_timeout=SUPABASE_TIMEOUT_SECONDS,
            ),
        )
    except ImportError:
        # Older supabase-py has no ClientOptions — fall back rather than
        # breaking startup.
        logger.warning("supabase.ClientOptions unavailable — no client timeout set")
        client = create_client(SUPABASE_URL, SUPABASE_KEY)

    _local.client = client
    return client


def _exec(query):
    """
    Runs a Supabase query, retrying dropped connections.

    Every call site goes through this. Retries are safe here because the
    operations are idempotent in practice: reads obviously so, and the writes
    are upserts or inserts that get retried only when no response arrived at
    all — so the first attempt almost certainly never reached the server.
    """
    last_error = None
    for attempt in range(1, SUPABASE_MAX_ATTEMPTS + 1):
        try:
            return query.execute()
        except Exception as e:
            last_error = e
            if attempt < SUPABASE_MAX_ATTEMPTS:
                logger.warning(
                    "Supabase → attempt %d/%d failed (%s) — retrying",
                    attempt, SUPABASE_MAX_ATTEMPTS, e,
                )
    raise last_error


def check_supabase_connection() -> bool:
    """Pings Supabase on startup so connection issues surface immediately."""
    try:
        client = get_supabase()
        _exec(client.table("sessions").select("id").limit(1))
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
        result = _exec(client.table("sessions").insert({
            "organization_id": organization_id
        }))
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
        result = _exec(client.table("sessions") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .order("created_at", desc=True) \
            )
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
        _exec(client.table("sessions").delete().eq("id", session_id))
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

        result = _exec(client.table("messages").insert(payload))
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
        result = _exec(client.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            )
        return result.data
    except Exception as e:
        logger.error("Supabase → get_messages failed: %s", e)
        return []



# ── Collections ───────────────────────────────────────────────────────────────

def get_collections_from_db(organization_id: str = "matz-demo-org") -> list:
    """Returns all collections from Supabase."""
    try:
        client = get_supabase()
        result = _exec(client.table("collections") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .order("created_at", desc=False) \
            )
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
        result = _exec(client.table("collections").insert({
            "name":            name,
            "description":     description,
            "icon":            icon,
            "organization_id": organization_id,
        }))
        logger.info("Supabase → collection created: %s", name)
        return result.data[0]
    except Exception as e:
        logger.error("Supabase → create_collection failed: %s", e)
        return None



def delete_collection_from_db(collection_id: str) -> bool:
    """Deletes a collection from Supabase."""
    try:
        client = get_supabase()
        _exec(client.table("collections").delete().eq("id", collection_id))
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
        result = _exec(client.table("workspace_settings") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .limit(1))
        if result.data:
            return result.data[0]
        return {"organization_id": organization_id, "name": ""}
    except Exception as e:
        logger.error("Supabase → get_workspace_settings failed: %s", e)
        return {"organization_id": organization_id, "name": ""}


def upsert_workspace_settings(organization_id: str, name: str) -> dict:
    """Creates or updates workspace settings for an organization."""
    try:
        client = get_supabase()
        result = _exec(client.table("workspace_settings").upsert({
            "organization_id": organization_id,
            "name": name,
        }))
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
        result = _exec(client.table("assistant_config") \
            .select("*") \
            .eq("organization_id", organization_id) \
            .limit(1))
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
        result = _exec(client.table("assistant_config").upsert({
            "organization_id": organization_id,
            "name": name,
            "personality": personality,
            "instructions": instructions,
        }))
        logger.info("Supabase → assistant config saved: %s", organization_id)
        return result.data[0]
    except Exception as e:
        logger.error("Supabase → upsert_assistant_config failed: %s", e)
        return None



# ── Usage ─────────────────────────────────────────────────────────────────────

def count_user_questions(organization_id: str = "matz-demo-org") -> int:
    """
    Counts user messages across all sessions for an organization.

    Uses a database-side join (see perf_fix.sql). The previous version fetched
    every session id and passed them to .in_() — PostgREST serialises that into
    the URL, so at a few hundred sessions the request took minutes and held a
    connection open the whole time.
    """
    try:
        client = get_supabase()
        result = _exec(client.rpc("count_org_user_questions", {"org_id": organization_id}))
        return int(result.data or 0)
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
        "success_rate_change_pct": None,
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

        # One server-side join instead of fetching every session id and
        # passing them to .in_() — see count_user_questions for why that was
        # slow enough to stall the page.
        messages = _exec(client.rpc("org_messages_since", {
            "org_id": organization_id,
            "since": previous_start.isoformat(),
        })).data or []

        if not messages:
            return _empty_analytics(organization_id, days)

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

        # Same rate for the preceding period, so the UI can show a trend.
        # Expressed in PERCENTAGE POINTS (e.g. 94.8 vs 91.7 -> +3.1), not as a
        # percent change of a percent.
        previous_checked = [
            m for m in previous_msgs
            if m["role"] == "assistant" and m.get("is_grounded") is not None
        ]
        previous_rate = (
            sum(1 for m in previous_checked if m["is_grounded"]) / len(previous_checked) * 100
            if previous_checked else None
        )
        success_rate_change_pct = (
            round(success_rate_pct - previous_rate, 1)
            if success_rate_pct is not None and previous_rate is not None
            else None
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
            "success_rate_change_pct": success_rate_change_pct,
            "avg_response_time_ms": avg_response_time_ms,
            "avg_response_time_change_ms": avg_response_time_change_ms,
            "questions_over_time": questions_over_time,
            "popular_collections": popular_collections,
            "organization_id": organization_id,
        }
    except Exception as e:
        logger.error("Supabase → get_analytics_summary failed: %s", e)
        return _empty_analytics(organization_id, days)


# ==========================================================================
# config_cache
# ==========================================================================

"""
src/utils/config_cache.py
--------------------------
Short-lived in-process cache for the workspace assistant config.

The config is needed on every chat turn to build the system prompt. Hitting
Supabase each time would add a network round-trip to every message, so it is
cached here and explicitly invalidated the moment an admin saves new settings
(see api/controllers/settings_controller.py) — that gives instant reflection in
the assistant without a per-request query.

The TTL is a safety net for the case where the config is changed outside this
process (directly in Supabase, or by a second API worker).

Note: this cache is per-process. Running uvicorn with multiple workers means
each worker holds its own copy, and a save only invalidates the worker that
handled it — the others catch up within the TTL. If you move to multiple
workers and need instant consistency, replace this with Redis or drop the
cache and query per request.
"""

import time


_TTL_SECONDS = 60

# organization_id -> (fetched_at_epoch, config_dict)
_cache: dict[str, tuple[float, dict]] = {}


def get_cached_assistant_config(organization_id: str = "matz-demo-org") -> dict:
    entry = _cache.get(organization_id)
    if entry and (time.time() - entry[0]) < _TTL_SECONDS:
        return entry[1]

    config = get_assistant_config(organization_id)
    _cache[organization_id] = (time.time(), config)
    return config


def invalidate_assistant_config(organization_id: str | None = None) -> None:
    """Called after an admin saves, so the next message uses the new config."""
    if organization_id is None:
        _cache.clear()
        logger.info("Assistant config cache → cleared (all orgs)")
    else:
        _cache.pop(organization_id, None)
        logger.info("Assistant config cache → invalidated: %s", organization_id)
