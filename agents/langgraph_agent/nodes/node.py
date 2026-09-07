"""LangGraph node implementations (retrieve / generate / safety)."""

# ==========================================================================
# retrieve
# ==========================================================================

"""
src/nodes/retrieve.py
----------------------
Two retrieval functions:
1. retrieve_context  — semantic search for specific questions (existing)
2. retrieve_all      — fetch ALL chunks for overview/summary questions (new)

Maps to ERD:
- organization_id → organizations.id (tenant isolation)
- chunk content   → document_chunks.content
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest

from agents.langgraph_agent.models.models import AgentState
from agents.langgraph_agent.embeddings.embeddings import embed_text
from agents.langgraph_agent.utils.utils import correct_query
from agents.langgraph_agent.utils.utils import logger

load_dotenv()

QDRANT_URL      = os.environ.get("QDRANT_URL")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY")
COLLECTION      = "matz_chunks"
TOP_K           = 3
ORGANIZATION_ID = "matz-demo-org"
MIN_SCORE       = 0.1

_qdrant_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def retrieve_context(state: AgentState) -> AgentState:
    """
    Semantic search — finds top K most similar chunks for specific questions.
    Used for all knowledge queries except overview.
    """
    query  = correct_query(state["user_query"])
    state  = {**state, "user_query": query}
    client = _get_client()

    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=ORGANIZATION_ID),
                )
            ]
        ),
        limit=TOP_K,
        with_payload=True,
    ).points

    docs = []
    for hit in results:
        payload = hit.payload
        score   = round(hit.score, 3)

        if score < MIN_SCORE:
            continue

        formatted = (
            f"[Source: {payload['document_title']} · "
            f"{payload['collection_name']} · "
            f"Page {payload['page_number']} | "
            f"Relevance: {score}]\n"
            f"{payload['content']}"
        )
        docs.append(formatted)

    return {**state, "retrieved_docs": docs}


def retrieve_all(state: AgentState) -> AgentState:
    """
    Fetches ALL chunks for this organization from Qdrant.
    Used for overview/summary questions like:
    - "what are all the policies?"
    - "what can you help me with?"
    - "give me a summary of everything"

    When new documents are uploaded and embedded, they are automatically
    included in the response — no code changes needed.
    """
    client = _get_client()
    logger.info("Retrieve all → fetching all chunks for org: %s", ORGANIZATION_ID)

    # Scroll through ALL points for this organization
    all_docs = []
    offset   = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="organization_id",
                        match=MatchValue(value=ORGANIZATION_ID),
                    )
                ]
            ),
            limit=100,          # fetch 100 at a time
            offset=offset,
            with_payload=True,
            with_vectors=False, # no vectors needed for overview
        )

        for point in results:
            payload = point.payload
            formatted = (
                f"[Source: {payload['document_title']} · "
                f"{payload['collection_name']} · "
                f"Page {payload['page_number']}]\n"
                f"{payload['content']}"
            )
            all_docs.append(formatted)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("Retrieve all → fetched %d chunks", len(all_docs))
    return {**state, "retrieved_docs": all_docs, "graded_docs": all_docs}


# ==========================================================================
# generate
# ==========================================================================

"""
src/nodes/generate.py
----------------------
Generates the final answer using the Groq LLM with multi-turn memory.
Includes chat history summarization to keep token usage low.

Cases handled:
  0a: injection    → fixed block response, no LLM
  0b: emergency    → fixed urgent response, no LLM
  1:  overview     → LLM summarizes all chunks into structured list
  2:  graded docs  → LLM answers using specific chunks
  3:  small_talk / out_of_scope → natural LLM, no context
  4:  no docs + history → LLM checks history only
  5:  no docs + no history → fixed "not found", no LLM
"""

from agents.langgraph_agent.models.models import AgentState
from agents.langgraph_agent.LLMs.llm import call_llm
from agents.langgraph_agent.prompts.prompt import (
    build_system_prompt, EMERGENCY_RESPONSE, NOT_FOUND_RESPONSE, injection_response,
)
from agents.langgraph_agent.utils.utils import maybe_summarize
from agents.langgraph_agent.utils.utils import logger

MAX_HISTORY_MESSAGES = 20


def generate_answer(state: AgentState) -> AgentState:
    query_type = state.get("query_type", "knowledge")

    # Built fresh each turn from the workspace's saved Settings → Assistant
    # values, so admin changes take effect on the very next message.
    config = state.get("assistant_config") or {}
    system_prompt = build_system_prompt(config)

    # ── Case 0a: Injection ────────────────────────────────────────────────────
    if query_type == "injection":
        answer = injection_response(config.get("name"))
        updated_history = _append_turn(
            state["chat_history"], state["user_query"], answer
        )
        return {
            **state, "answer": answer, "chat_history": updated_history,
            "is_fixed_response": True,
        }

    # ── Case 0b: Emergency ────────────────────────────────────────────────────
    if state["is_emergency"]:
        answer = EMERGENCY_RESPONSE
        updated_history = _append_turn(
            state["chat_history"], state["user_query"], answer
        )
        return {
            **state, "answer": answer, "chat_history": updated_history,
            "is_fixed_response": True,
        }

    # ── Summarize history if too long ─────────────────────────────────────────
    # This runs before building messages_for_llm so the LLM always
    # gets a compact history regardless of conversation length
    history = maybe_summarize(state["chat_history"])

    docs = state.get("graded_docs") or []

    if docs and query_type == "overview":
        # ── Case 1: Overview — summarize ALL chunks ───────────────────────────
        context = "\n\n".join(docs)
        current_message = (
            f"COMPANY KNOWLEDGE BASE — ALL AVAILABLE DOCUMENTS:\n{context}\n\n"
            f"USER REQUEST:\n{state['user_query']}\n\n"
            f"INSTRUCTION: Based on ALL the documents above, provide a clear structured "
            f"summary of all available company policies and guidelines. "
            f"List each policy with its key points. "
            f"Do not leave any document out. "
            f"Format as a numbered list with the document name as heading and 2-3 key points under each."
        )
        messages_for_llm = history + [{"role": "user", "content": current_message}]
        try:
            # Overview answers summarize every document in the knowledge base,
            # which can easily exceed the default 1024-token cap and get cut
            # off mid-list — give this case a much larger budget.
            answer = call_llm(system_prompt=system_prompt, messages=messages_for_llm, max_tokens=4096)
        except RuntimeError as e:
            answer = f"[Setup needed] {e}"

    elif docs:
        # ── Case 2: Specific chunks → answer specific question ────────────────
        context = "\n".join(docs)
        current_message = (
            f"CONTEXT:\n{context}\n\n"
            f"USER MESSAGE:\n{state['user_query']}\n\n"
            f"[LANGUAGE INSTRUCTION: Detect the language of the USER MESSAGE only. "
            f"If it has typos but is otherwise English, reply in English. "
            f"Reply in Roman Urdu ONLY if it clearly contains Urdu words like "
            f"'kya', 'hai', 'batao', 'mujhe'.]"
        )
        messages_for_llm = history + [{"role": "user", "content": current_message}]
        try:
            answer = call_llm(system_prompt=system_prompt, messages=messages_for_llm)
        except RuntimeError as e:
            answer = f"[Setup needed] {e}"

    elif query_type in ("small_talk", "out_of_scope"):
        # ── Case 3: Small talk or out of scope → natural LLM ─────────────────
        current_message = f"USER MESSAGE:\n{state['user_query']}"
        messages_for_llm = history + [{"role": "user", "content": current_message}]
        try:
            answer = call_llm(system_prompt=system_prompt, messages=messages_for_llm)
        except RuntimeError as e:
            answer = f"[Setup needed] {e}"

    elif history:
        # ── Case 4: No chunks + history → check history only ─────────────────
        current_message = (
            f"USER MESSAGE:\n{state['user_query']}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Check the CONVERSATION HISTORY above carefully.\n"
            f"2. If this question was already answered in a previous turn, "
            f"use that answer and reference it naturally.\n"
            f"3. If the answer is NOT in the history, respond with EXACTLY:\n"
            f"   'I could not find this in the available documents. "
            f"Please check with the relevant team.'\n"
            f"4. NEVER use your own training knowledge.\n"
            f"5. NEVER fabricate any policy, number, or rule not already in the history.\n"
            f"6. Your response must be ONLY the exact sentence in step 3 "
            f"if the answer is not in history — nothing more.\n"
            f"[LANGUAGE INSTRUCTION: Reply in the same language as the USER MESSAGE.]"
        )
        messages_for_llm = history + [{"role": "user", "content": current_message}]
        try:
            answer = call_llm(system_prompt=system_prompt, messages=messages_for_llm)
        except RuntimeError as e:
            answer = f"[Setup needed] {e}"

    else:
        # ── Case 5: No chunks + no history → fixed response, no LLM ─────────
        answer = NOT_FOUND_RESPONSE
        updated_history = _append_turn(
            state["chat_history"], state["user_query"], answer
        )
        return {
            **state, "answer": answer, "chat_history": updated_history,
            "is_fixed_response": True,
        }

    # Always append to ORIGINAL history (not summarized) to preserve full turns
    updated_history = _append_turn(
        state["chat_history"], state["user_query"], answer
    )
    return {**state, "answer": answer, "chat_history": updated_history}


def _append_turn(history: list, user_query: str, answer: str) -> list:
    """Adds the latest turn to history, trimmed to last N messages."""
    new_history = history + [
        {"role": "user",      "content": user_query},
        {"role": "assistant", "content": answer},
    ]
    return new_history[-MAX_HISTORY_MESSAGES:]


# ==========================================================================
# safety
# ==========================================================================

"""
src/nodes/safety.py
--------------------
Flags urgent/emergency queries so generate_answer can short-circuit
to a fixed "contact the relevant team immediately" response.
This avoids letting the LLM handle sensitive incidents unpredictably.
"""

from agents.langgraph_agent.models.models import AgentState

URGENT_KEYWORDS = [
    # IT & Security incidents
    "data breach", "hacked", "hack", "ransomware", "malware",
    "unauthorized access", "credentials leaked", "password stolen",
    "phishing", "suspicious email", "suspicious link",
    "server is down", "server down", "system down", "system is down",
    "production down", "production is down",
    "critical bug", "outage",

    # HR urgent
    "harassment", "discrimination", "misconduct", "fired", "terminated",

    # Legal / compliance
    "lawsuit", "legal action", "regulatory breach", "audit failure",
]


def safety_check(state: AgentState) -> AgentState:
    """Set is_emergency=True if the query matches any urgent keyword."""
    query = state["user_query"].lower()
    is_urgent = any(kw in query for kw in URGENT_KEYWORDS)
    return {**state, "is_emergency": is_urgent}
