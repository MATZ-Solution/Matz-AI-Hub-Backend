"""api/routes/health_routes.py — root + health check, not tied to any page."""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agents.langgraph_agent.utils.utils import logger
from api.config.settings import (
    APP_TITLE, GROQ_MODEL_FAST, HEALTH_PROVIDER_CACHE_SECONDS,
)

router = APIRouter(tags=["health"])

# (checked_at, ok) — see HEALTH_PROVIDER_CACHE_SECONDS for why this is cached.
_provider_cache: tuple[float, bool] | None = None


def _check_provider() -> bool:
    """A real 1-token completion, so an expired or revoked key shows up here."""
    global _provider_cache
    now = time.time()
    if _provider_cache and now - _provider_cache[0] < HEALTH_PROVIDER_CACHE_SECONDS:
        return _provider_cache[1]
    try:
        from agents.langgraph_agent.LLMs.llm import ping_provider
        ping_provider(GROQ_MODEL_FAST)
        ok = True
    except Exception as e:  # never surfaced to the caller, only logged
        logger.error("Health → provider check failed: %s", e)
        ok = False
    _provider_cache = (now, ok)
    return ok


def _check_qdrant() -> bool:
    try:
        from data import qdrant_store
        qdrant_store.ping()
        return True
    except Exception as e:
        logger.error("Health → Qdrant check failed: %s", e)
        return False


@router.get("/")
def root():
    return {"message": f"{APP_TITLE} API", "docs": "/docs"}


@router.get("/health")
def health_check():
    """{ok, provider, qdrant}. Returns 503 when any dependency is down, so
    uptime monitors and Fly's health check see the failure."""
    provider = _check_provider()
    qdrant = _check_qdrant()
    ok = provider and qdrant
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ok": ok, "provider": provider, "qdrant": qdrant},
    )
