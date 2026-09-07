"""
src/models/llm_client.py
-------------------------
Thin wrapper around Groq's chat completion API (free & fast).

Setup:
    1. pip install groq python-dotenv
    2. Get a free API key: https://console.groq.com/keys
    3. Add to your .env file:
         GROQ_API_KEY=your-key-here
       (Never commit this file to git — it's already in .gitignore)

Available models on Groq (as of 2026 — check https://console.groq.com/docs/models
for the current list, since Groq deprecates models periodically):
    - openai/gpt-oss-120b                       → best quality, production default
    - openai/gpt-oss-20b                        → smaller/faster
    - meta-llama/llama-4-scout-17b-16e-instruct → supports vision, long context
    - qwen/qwen3-32b                            → reasoning model
"""

import os
from groq import Groq
from dotenv import load_dotenv
from agents.langgraph_agent.utils.utils import logger

load_dotenv()

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily initialise the Groq client (once per process)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at "
                "https://console.groq.com/keys then add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    system_prompt: str,
    messages: list,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """
    Send a conversation to the LLM and return the assistant reply.

    Args:
        system_prompt:     The system instruction string.
        messages:          List of {"role": "user"|"assistant", "content": "..."}
                            representing the conversation so far (oldest first).
                            Do NOT include the system message here.
        model:              Groq model ID.
        temperature:        Sampling temperature (0 = deterministic).
        max_tokens:         Maximum tokens in the reply.
        response_format:    Optional, e.g. {"type": "json_object"} to force
                             valid-JSON-only output (reasoning models on Groq
                             can otherwise wander into free text before JSON).
        reasoning_effort:   Optional, "low" | "medium" | "high" — only applies
                             to reasoning-capable models (e.g. gpt-oss). Use
                             "low" for small structured tasks (classification,
                             JSON verdicts) so the model doesn't burn its
                             token budget on hidden reasoning.

    Returns:
        The assistant's reply as a plain string.
    """
    client = _get_client()
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    kwargs = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning(
            "LLM response TRUNCATED — hit max_tokens=%d (model=%s). "
            "Consider raising max_tokens for this call site.",
            max_tokens, model,
        )
    return choice.message.content