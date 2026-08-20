"""
src/nodes/tools/hallucination_checker.py
-----------------------------------------
Hallucination Checker node — sits between generate_answer and citation_builder.

What it does:
1. Takes the generated answer and the graded chunks
2. Asks the LLM: "Is this answer supported by the chunks?"
3. If grounded → pass to citation_builder
4. If not grounded → regenerate with stricter prompt (max 2 retries)
5. After max retries → return best answer available

Maps to ERD:
- graded_docs          → document_chunks.content (source of truth)
- answer               → messages.content (what we are verifying)
- generation_attempts  → retry counter (max 2)
"""

import json
from agent.src.models.state import AgentState
from agent.src.models.llm_client import call_llm
from agent.src.prompts.system_prompt import SYSTEM_PROMPT
from agent.src.utils.logger import logger

MAX_GENERATION_ATTEMPTS = 2

# ── Hallucination check prompt ────────────────────────────────────────────────
HALLUCINATION_PROMPT = (
    "You are a hallucination checker for a company knowledge assistant.\n\n"
    "Given a generated answer and the source chunks it was based on, "
    "determine if the answer is grounded in the chunks.\n\n"
    "Respond ONLY with a JSON object — no explanation, no markdown:\n"
    "{\n"
    '  "is_grounded": true | false,\n'
    '  "reason": "one short sentence"\n'
    "}\n\n"
    "Rules:\n"
    "- is_grounded = true if every claim in the answer can be found in the chunks\n"
    "- is_grounded = false if the answer contains ANY information not in the chunks\n"
    "- Minor rewording is fine — judge the facts, not the exact words\n"
    "- If the answer says 'I could not find this' → always true (no claims made)\n"
    "- If the answer is a greeting or small talk → always true\n"
    "- Be strict — any fabricated number, date, or policy detail = false"
)

# ── Strict regeneration prompt ────────────────────────────────────────────────
STRICT_REGEN_PROMPT = (
    "You are MATZ Assistant, an AI knowledge assistant for the organization.\n\n"
    "CRITICAL: Your previous answer contained information not found in the source documents. "
    "You must regenerate the answer using ONLY the information in the CONTEXT provided.\n\n"
    "Rules:\n"
    "- Use ONLY facts explicitly stated in the CONTEXT\n"
    "- If the CONTEXT does not contain enough information, say: "
    "'I could not find this in the available documents. Please check with the relevant team.'\n"
    "- Do NOT add any information from your training knowledge\n"
    "- Do NOT guess or infer facts not explicitly in the CONTEXT\n"
    "- Always cite the source document and section"
)


def check_hallucination(state: AgentState) -> AgentState:
    """
    Verifies the generated answer is grounded in the retrieved chunks.
    Triggers regeneration if hallucination detected.
    """
    answer   = state.get("answer", "")
    chunks   = state.get("graded_docs") or []
    attempts = state.get("generation_attempts", 0)

    # Skip check if no chunks were used — answer came from history or fixed response
    if not chunks:
        logger.info("Hallucination checker → no chunks used, skipping check")
        return {**state, "is_grounded": True, "generation_attempts": attempts}

    # Skip check for fixed responses
    if answer in (
        "I could not find this in the available documents. Please check with the relevant team.",
        "This appears to be an urgent matter. Please contact your manager or the relevant department immediately. For IT security incidents, contact the security team right away.",
        "I'm not able to share that information. I'm MATZ Assistant, here to help you find answers from company documents. How can I help you today?"
    ):
        logger.info("Hallucination checker → fixed response, skipping check")
        return {**state, "is_grounded": True, "generation_attempts": attempts}

    is_grounded, reason = _check_grounding(answer, chunks)
    logger.info("Hallucination checker → grounded: %s | %s", is_grounded, reason)

    if is_grounded:
        return {**state, "is_grounded": True, "generation_attempts": attempts}

    # Not grounded → regenerate if retries available
    if attempts < MAX_GENERATION_ATTEMPTS:
        logger.info(
            "Hallucination checker → regenerating (attempt %d/%d)",
            attempts + 1, MAX_GENERATION_ATTEMPTS
        )
        context = "\n".join(chunks)
        new_answer = _regenerate_strict(state["user_query"], context)
        return {
            **state,
            "answer":              new_answer,
            "is_grounded":         False,
            "generation_attempts": attempts + 1,
        }

    # Max retries reached — return last answer
    logger.warning("Hallucination checker → max retries reached, returning last answer")
    return {**state, "is_grounded": True, "generation_attempts": attempts}


def _check_grounding(answer: str, chunks: list) -> tuple:
    """Ask LLM if the answer is grounded in the chunks."""
    try:
        context = "\n".join(chunks)
        message = (
            f"SOURCE CHUNKS:\n{context}\n\n"
            f"GENERATED ANSWER:\n{answer}"
        )
        response = call_llm(
            system_prompt=HALLUCINATION_PROMPT,
            messages=[{"role": "user", "content": message}],
            model="openai/gpt-oss-120b",
            temperature=0.0,
            max_tokens=80,
        )
        clean  = response.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        return bool(result.get("is_grounded", True)), result.get("reason", "")

    except Exception as e:
        logger.warning("Hallucination check failed (%s) — assuming grounded", e)
        return True, "check failed — assumed grounded"


def _regenerate_strict(query: str, context: str) -> str:
    """Regenerate answer with stricter prompt that forbids training knowledge."""
    try:
        message = (
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"Answer using ONLY the information in the CONTEXT above. "
            f"Do not use any other knowledge."
        )
        response = call_llm(
            system_prompt=STRICT_REGEN_PROMPT,
            messages=[{"role": "user", "content": message}],
            temperature=0.1,
            max_tokens=500,
        )
        return response.strip()

    except Exception as e:
        logger.warning("Strict regeneration failed (%s)", e)
        return (
            "I could not find this in the available documents. "
            "Please check with the relevant team."
        )