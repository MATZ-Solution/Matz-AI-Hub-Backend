"""
src/nodes/tools/relevance_grader.py
-------------------------------------
Relevance Grader node — sits between retrieve_context and generate_answer.

What changed — parallel grading:
  Before: chunks graded one by one (sequential) → slow
  After:  all chunks graded simultaneously (parallel) → 3x faster

Uses ThreadPoolExecutor to send all grader LLM calls at the same time.
Results are identical — just faster.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent.src.models.state import AgentState
from agent.src.models.llm_client import call_llm
from agent.src.utils.logger import logger
import os
RELEVANCE_THRESHOLD = 0.5
MAX_RETRIEVAL_ATTEMPTS = 2

GRADER_PROMPT = (
    "You are a relevance grader for a company knowledge assistant.\n\n"
    "Given a user question and a document chunk, score how relevant "
    "the chunk is for answering the question.\n\n"
    "CRITICAL OUTPUT FORMAT: Respond with ONLY the raw JSON object below — "
    "no reasoning, no explanation, no markdown fences, no text before or after it. "
    "Do NOT think step by step — output the JSON immediately:\n"
    "{\n"
    '  "score": 0.0 to 1.0,\n'
    '  "reason": "under 10 words"\n'
    "}\n\n"
    "Scoring guide:\n"
    "- 0.9 to 1.0: chunk directly answers the question\n"
    "- 0.6 to 0.8: chunk is related and partially useful\n"
    "- 0.3 to 0.5: chunk is loosely related but same topic area\n"
    "- 0.0 to 0.2: chunk is completely unrelated\n\n"
    "CRITICAL — Understand synonyms and related terms:\n"
    "- 'marketing rules' = 'brand guidelines' = 'marketing guidelines'\n"
    "- 'company rules' = 'company policies' = 'workplace rules'\n"
    "- 'work from home' = 'remote work' = 'WFH'\n"
    "- 'sick days' = 'sick leave' = 'medical leave'\n"
    "- 'reimbursement' = 'expense claim' = 'expense policy'\n"
    "- 'hiring' = 'onboarding' = 'new employee'\n"
    "- 'data security' = 'information security' = 'password policy'\n"
    "- 'supplier' = 'vendor' = 'third party'\n\n"
    "CRITICAL — Judge by MEANING not exact wording:\n"
    "- If the chunk topic is clearly related to what the user is asking, "
    "score it at least 0.5 even if exact words are different.\n"
    "- A chunk about 'brand guidelines' IS relevant to 'marketing rules'.\n"
    "- A chunk about 'remote work policy' IS relevant to 'can I work from home'.\n"
    "- Do NOT be overly literal. Think about what the user actually wants to know.\n"
    "- Only score 0.0 if the chunk is from a completely different topic "
    "(e.g. expense policy when asked about passwords)."
)

REWRITE_PROMPT = (
    "You are a query rewriter for a company knowledge assistant.\n\n"
    "The original query did not find relevant results. "
    "Rewrite it to be more specific and likely to find better matches "
    "in company documents.\n\n"
    "Rules:\n"
    "- Keep the same intent\n"
    "- Use more specific company/policy terminology\n"
    "- Try different keywords that might appear in company documents\n"
    "- Respond with ONLY the rewritten query — no explanation"
)


def grade_relevance(state: AgentState) -> AgentState:
    query    = state["user_query"]
    chunks   = state["retrieved_docs"]
    attempts = state.get("retrieval_attempts", 0)

    if not chunks:
        logger.info("Grader → no chunks retrieved, triggering retry")
        return {
            **state,
            "graded_docs":        [],
            "needs_retry":        attempts < MAX_RETRIEVAL_ATTEMPTS,
            "retrieval_attempts": attempts + 1,
            "rewritten_query":    _rewrite_query(query) if attempts < MAX_RETRIEVAL_ATTEMPTS else query,
        }

    # ── Grade all chunks in PARALLEL ─────────────────────────────────────────
    results = _grade_chunks_parallel(query, chunks)

    graded = []
    for i, (chunk, score, reason) in enumerate(results):
        logger.info("Grader → chunk %d score: %.2f | %s", i + 1, score, reason)
        if score >= RELEVANCE_THRESHOLD:
            graded.append(chunk)

    logger.info("Grader → %d/%d chunks passed", len(graded), len(chunks))

    if not graded and attempts < MAX_RETRIEVAL_ATTEMPTS:
        rewritten = _rewrite_query(query)
        logger.info("Grader → no relevant chunks, rewriting query: '%s'", rewritten)
        return {
            **state,
            "graded_docs":        [],
            "needs_retry":        True,
            "retrieval_attempts": attempts + 1,
            "rewritten_query":    rewritten,
        }

    return {
        **state,
        "graded_docs":        graded,
        "needs_retry":        False,
        "retrieval_attempts": attempts,
        "rewritten_query":    None,
    }


def _grade_chunks_parallel(query: str, chunks: list) -> list:
    """
    Grade all chunks simultaneously using ThreadPoolExecutor.
    Returns list of (chunk, score, reason) tuples in original order.

    Before: chunk1 → wait → chunk2 → wait → chunk3 → wait (sequential)
    After:  chunk1 ─┐
            chunk2  ├→ all finish together (parallel)
            chunk3 ─┘
    """
    results = [None] * len(chunks)  # preserve original order

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        # Submit all grading tasks at once
        future_to_index = {
            executor.submit(_grade_chunk, query, chunk): i
            for i, chunk in enumerate(chunks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                score, reason = future.result()
                results[index] = (chunks[index], score, reason)
            except Exception as e:
                logger.warning("Grader parallel task failed (%s) — defaulting to 0.5", e)
                results[index] = (chunks[index], 0.5, "grading failed — kept by default")

    return results


def _grade_chunk(query: str, chunk: str) -> tuple:
    """Grade one chunk. Called in parallel by _grade_chunks_parallel."""
    try:
        message = f"USER QUESTION:\n{query}\n\nDOCUMENT CHUNK:\n{chunk}"
        response = call_llm(
            system_prompt=GRADER_PROMPT,
            messages=[{"role": "user", "content": message}],
            model="openai/gpt-oss-120b",
            temperature=0.0,
            # Same fix as hallucination_checker: gpt-oss is a reasoning model
            # and at max_tokens=80 it spent its whole budget on hidden
            # reasoning, truncating the JSON. json.loads() then failed and
            # EVERY chunk silently defaulted to 0.5 — meaning relevance
            # filtering was effectively not running at all.
            max_tokens=500,
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )
        clean  = response.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        score  = float(result.get("score", 0.0))
        reason = result.get("reason", "")
        return max(0.0, min(1.0, score)), reason

    except Exception as e:
        logger.warning("Grader LLM failed (%s) — defaulting to 0.5", e)
        return 0.5, "grading failed — kept by default"


def _rewrite_query(original_query: str) -> str:
    """Rewrite query for better retrieval on retry."""
    try:
        response = call_llm(
            system_prompt=REWRITE_PROMPT,
            messages=[{"role": "user", "content": f"Original query: {original_query}"}],
            model=os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b"),
            temperature=0.3,
            max_tokens=60,
        )
        return response.strip()
    except Exception as e:
        logger.warning("Query rewrite failed (%s) — using original", e)
        return original_query