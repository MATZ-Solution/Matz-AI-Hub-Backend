# src package
"""
src/nodes/router.py
--------------------
Router node — the first node in the MATZ agent graph.

Reads the user query and sets three fields in AgentState:
  - query_type:         "knowledge" | "small_talk" | "urgent" | "out_of_scope"
  - target_collection:  collection name to scope search, or None for all
  - search_strategy:    "semantic" | "keyword" | "hybrid"

This lets downstream nodes skip unnecessary work:
  - small_talk   → skip Qdrant entirely, go straight to generator
  - urgent       → go to safety_check immediately
  - knowledge    → go to Qdrant with correct collection scope
  - out_of_scope → go straight to generator with no context

Maps to ERD:
  - target_collection → knowledge_collections.name
  - organization_id   → organizations.id (tenant isolation, always applied)
"""

from agents.langgraph_agent.models.models import AgentState
from agents.langgraph_agent.LLMs.llm import call_llm
from agents.langgraph_agent.utils.utils import logger
import os
# ── Collection names — must match knowledge_collections.name in ERD ───────────
COLLECTIONS = [
    "HR Policies",
    "Security",
    "Operations",
    "Compliance",
    "Sales Enablement",
    "Marketing",
]

# ── Urgent keywords — same as safety.py ──────────────────────────────────────
# Router checks these FIRST before calling LLM — fast path for emergencies
URGENT_KEYWORDS = [
    "data breach", "hacked", "hack", "ransomware", "malware",
    "unauthorized access", "credentials leaked", "password stolen",
    "phishing", "suspicious email", "suspicious link",
    "server is down", "server down", "system down", "system is down",
    "production down", "production is down",
    "critical bug", "outage",
    "harassment", "discrimination", "misconduct",
    "lawsuit", "legal action", "regulatory breach", "audit failure",
]

# ── Small talk patterns — fast path, no LLM needed ───────────────────────────
SMALL_TALK_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "who are you", "what are you", "what can you do",
    "thank you", "thanks", "bye", "goodbye", "ok", "okay", "great",
]

# ── Router system prompt ──────────────────────────────────────────────────────
ROUTER_PROMPT = (
    "You are a query router for MATZ, a company AI knowledge assistant.\n\n"
    "Given a user query, respond ONLY with a JSON object — no explanation, no markdown.\n\n"
    "JSON format:\n"
    "{\n"
    '  "query_type": "knowledge" | "small_talk" | "out_of_scope",\n'
    '  "target_collection": "HR Policies" | "Security" | "Operations" | '
    '"Compliance" | "Sales Enablement" | "Marketing" | null,\n'
    '  "search_strategy": "semantic" | "keyword" | "hybrid"\n'
    "}\n\n"
    "Rules:\n"
    "- query_type = 'knowledge' if the question is about company policies, "
    "documents, procedures, guidelines, or any work-related topic.\n"
    "- query_type = 'small_talk' if it is a greeting, thanks, or casual chat.\n"
    "- query_type = 'out_of_scope' if it has nothing to do with company knowledge "
    "(e.g. general world facts, personal questions).\n"
    "- target_collection: pick the most relevant collection or null if the question "
    "spans multiple collections or is unclear.\n"
    "  * HR Policies     → leave, remote work, onboarding, benefits, attendance\n"
    "  * Security        → passwords, MFA, data protection, access control\n"
    "  * Operations      → expenses, travel, SOPs, processes\n"
    "  * Compliance      → vendors, audits, regulatory, procurement\n"
    "  * Sales Enablement → sales, discounts, proposals, quotes, pricing\n"
    "  * Marketing       → brand, logo, colors, campaigns, messaging\n"
    "- search_strategy:\n"
    "  * semantic  → default for most natural language questions\n"
    "  * keyword   → when the query contains exact section numbers, codes, or IDs\n"
    "  * hybrid    → when the query is complex and covers multiple topics\n"
    "Respond with ONLY the JSON object. No markdown, no explanation."
)


def route_query(state: AgentState) -> AgentState:
    """
    Classifies the user query and sets routing fields in state.
    Uses fast keyword checks first, LLM only when needed.
    """
    query = state["user_query"].lower().strip()

    # ── Fast path 1: Urgent keywords ─────────────────────────────────────────
    if any(kw in query for kw in URGENT_KEYWORDS):
        logger.info("Router → urgent (keyword match)")
        return {
            **state,
            "query_type": "urgent",
            "target_collection": None,
            "search_strategy": "semantic",
        }

    # ── Fast path 2: Small talk ───────────────────────────────────────────────
    if any(query == pat or query.startswith(pat) for pat in SMALL_TALK_PATTERNS):
        logger.info("Router → small_talk (pattern match)")
        return {
            **state,
            "query_type": "small_talk",
            "target_collection": None,
            "search_strategy": "semantic",
        }

    # ── LLM classification for everything else ────────────────────────────────
    try:
        import json
        response = call_llm(
            system_prompt=ROUTER_PROMPT,
            messages=[{"role": "user", "content": state["user_query"]}],
            model=os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b"),   # use fast model for routing
            temperature=0.0,
            max_tokens=100,
        )

        # Strip markdown fences if present
        clean = response.strip().strip("```json").strip("```").strip()
        routing = json.loads(clean)

        query_type        = routing.get("query_type", "knowledge")
        target_collection = routing.get("target_collection", None)
        search_strategy   = routing.get("search_strategy", "semantic")

        # Validate values
        if query_type not in ("knowledge", "small_talk", "out_of_scope"):
            query_type = "knowledge"
        if target_collection not in COLLECTIONS:
            target_collection = None
        if search_strategy not in ("semantic", "keyword", "hybrid"):
            search_strategy = "semantic"

        logger.info(
            "Router → %s | collection: %s | strategy: %s",
            query_type, target_collection, search_strategy
        )

        return {
            **state,
            "query_type": query_type,
            "target_collection": target_collection,
            "search_strategy": search_strategy,
        }

    except Exception as e:
        # If LLM classification fails for any reason, default to full knowledge search
        logger.warning("Router LLM failed (%s) — defaulting to knowledge/semantic", e)
        return {
            **state,
            "query_type": "knowledge",
            "target_collection": None,
            "search_strategy": "semantic",
        }
