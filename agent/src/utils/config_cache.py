"""
src/utils/config_cache.py
--------------------------
Short-lived in-process cache for the workspace assistant config.

The config is needed on every chat turn to build the system prompt. Hitting
Supabase each time would add a network round-trip to every message, so it is
cached here and explicitly invalidated the moment an admin saves new settings
(see api/controllers/settings_controller.py) — that gives instant reflection in
the assistant without a per-request query.

The TTL is a safety net for the case where the config is changed outside this
process (directly in Supabase, or by a second API worker).

Note: this cache is per-process. Running uvicorn with multiple workers means
each worker holds its own copy, and a save only invalidates the worker that
handled it — the others catch up within the TTL. If you move to multiple
workers and need instant consistency, replace this with Redis or drop the
cache and query per request.
"""

import time

from agent.src.utils.supabase_client import get_assistant_config
from agent.src.utils.logger import logger

_TTL_SECONDS = 60

# organization_id -> (fetched_at_epoch, config_dict)
_cache: dict[str, tuple[float, dict]] = {}


def get_cached_assistant_config(organization_id: str = "matz-demo-org") -> dict:
    entry = _cache.get(organization_id)
    if entry and (time.time() - entry[0]) < _TTL_SECONDS:
        return entry[1]

    config = get_assistant_config(organization_id)
    _cache[organization_id] = (time.time(), config)
    return config


def invalidate_assistant_config(organization_id: str | None = None) -> None:
    """Called after an admin saves, so the next message uses the new config."""
    if organization_id is None:
        _cache.clear()
        logger.info("Assistant config cache → cleared (all orgs)")
    else:
        _cache.pop(organization_id, None)
        logger.info("Assistant config cache → invalidated: %s", organization_id)