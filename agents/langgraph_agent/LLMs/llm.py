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
import time

import groq
from groq import Groq
from dotenv import load_dotenv
from agents.langgraph_agent.utils.utils import logger

load_dotenv()

_client: Groq | None = None

# Task A2: retry provider 429s and 5xx twice (after 1 s, then 3 s) and give
# each attempt a hard 30 s timeout. The SDK's own retries are turned off so
# these are the only ones.
RETRY_DELAYS_SECONDS = (1, 3)
REQUEST_TIMEOUT_SECONDS = 30


class ProviderError(Exception):
    """The LLM provider failed. `kind` maps onto the API error contract:
    "provider_unavailable" | "rate_limited" | "timeout"."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (groq.APITimeoutError, groq.APIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return status == 429 or (status is not None and status >= 500)


def _to_provider_error(exc: Exception) -> ProviderError:
    if isinstance(exc, groq.APITimeoutError):
        return ProviderError("timeout", str(exc))
    if getattr(exc, "status_code", None) == 429:
        return ProviderError("rate_limited", str(exc))
    # Expired/invalid key (401), connection failures, 5xx and anything else.
    return ProviderError("provider_unavailable", str(exc))


def _create_completion(**kwargs):
    """chat.completions.create with the A2 retry schedule. Raises ProviderError."""
    client = _get_client()
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except groq.GroqError as exc:
            if attempt < len(RETRY_DELAYS_SECONDS) and _is_retryable(exc):
                delay = RETRY_DELAYS_SECONDS[attempt]
                logger.warning("LLM call failed (%s), retrying in %ds", type(exc).__name__, delay)
                time.sleep(delay)
                continue
            raise _to_provider_error(exc) from exc


def _get_client() -> Groq:
    """Lazily initialise the Groq client (once per process)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ProviderError("provider_unavailable", "GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
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
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    kwargs = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = _create_completion(
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


def ping_provider(model: str) -> None:
    """
    Smallest possible real request (1 output token), used by GET /health.
    Raises if the key is missing, expired or revoked, or the provider is down.
    A truncated reply is fine — the request itself succeeding is the signal.
    """
    _create_completion(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
