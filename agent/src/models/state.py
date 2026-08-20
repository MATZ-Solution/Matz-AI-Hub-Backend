"""
src/models/state.py
--------------------
Shared state that flows through every node in the graph.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    # ── Core ──────────────────────────────────────────────────────────────────
    user_query: str
    chat_history: List[dict]
    retrieved_docs: List[str]
    is_emergency: bool
    answer: Optional[str]

    # ── Router fields ─────────────────────────────────────────────────────────
    query_type: Optional[str]        # "knowledge"|"small_talk"|"urgent"|"out_of_scope"|"injection"
    target_collection: Optional[str] # knowledge_collections.name in ERD
    search_strategy: Optional[str]   # "semantic"|"keyword"|"hybrid"

    # ── Relevance grader fields ───────────────────────────────────────────────
    graded_docs: List[str]           # chunks that passed relevance grading
    needs_retry: bool                # True = go back to Qdrant
    retrieval_attempts: int          # retry counter (max 2)
    rewritten_query: Optional[str]   # reformulated query for retry

    # ── Hallucination checker fields ──────────────────────────────────────────
    is_grounded: bool                # True = answer verified against chunks
    generation_attempts: int         # regeneration counter (max 2)

    # ── Citation builder fields ───────────────────────────────────────────────
    citations: List[dict]            # list of Citation dicts → message_citations table