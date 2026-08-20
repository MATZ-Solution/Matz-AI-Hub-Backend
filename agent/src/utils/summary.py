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
from agent.src.utils.logger import logger

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
        from src.models.llm_client import call_llm

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
