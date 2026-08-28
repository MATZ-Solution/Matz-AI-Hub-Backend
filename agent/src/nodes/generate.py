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

from agent.src.models.state import AgentState
from agent.src.models.llm_client import call_llm
from agent.src.prompts.system_prompt import (
    build_system_prompt, EMERGENCY_RESPONSE, NOT_FOUND_RESPONSE, injection_response,
)
from agent.src.utils.summary import maybe_summarize
from agent.src.utils.logger import logger

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