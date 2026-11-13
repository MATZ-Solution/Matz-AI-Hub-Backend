"""
api/rate_limit.py
------------------
Abuse protection (task A5): sliding-window rate limits kept in memory.

Per user:  chat 20 / minute, upload 10 / hour, search 60 / minute.
Per IP:    120 / minute across the whole API (except /health).

The task list rules out Redis, so the windows live in this process. With
more than one API machine each would keep its own counts (see DECISIONS.md).
"""

import threading
import time
from collections import deque

from fastapi import Request

from api.errors import ApiError, ErrorCode, error_response
from agents.langgraph_agent.utils.utils import logger

# name → (max requests, window in seconds, message shown to the user)
USER_LIMITS: dict[str, tuple[int, float, str]] = {
    "chat": (20, 60, "You're sending messages too quickly. Wait a minute and try again."),
    "upload": (10, 3600, "You've reached the limit of 10 uploads an hour. Try again later."),
    "search": (60, 60, "Too many searches in a short time. Wait a moment and try again."),
}
IP_LIMIT = (120, 60)
IP_EXEMPT_PATHS = {"/health"}


class SlidingWindowLimiter:
    """Allows at most `limit` hits per key in any `window`-second span."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def hit(self, key: str, limit: int, window: float) -> bool:
        """Records a hit and returns True, or returns False if over the limit
        (a refused hit is not recorded, so a blocked client recovers on time)."""
        now = time.monotonic()
        with self._lock:
            self._sweep(now, window)
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def _sweep(self, now: float, window: float) -> None:
        # Drop idle keys now and then so memory doesn't grow with every IP seen.
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        longest = max(window, *(w for _, w, _ in USER_LIMITS.values()))
        for key in [k for k, h in self._hits.items() if not h or now - h[-1] >= longest]:
            del self._hits[key]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def check_user_limit(user_id: str, name: str) -> None:
    """Raises rate_limited if this user is over the named per-user limit."""
    limit, window, message = USER_LIMITS[name]
    if not limiter.hit(f"user:{name}:{user_id}", limit, window):
        logger.info("Rate limit → %s exceeded by user %s", name, user_id)
        raise ApiError(ErrorCode.rate_limited, message)


def client_ip(request: Request) -> str:
    """The caller's IP. Behind the web app's proxy and Fly, the browser's IP is
    the first X-Forwarded-For entry; without that header, the socket peer."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


def install_ip_rate_limit(app) -> None:
    """Per-IP limit on every route except /health. Install it after auth so it
    runs before auth: unauthenticated floods are limited too."""

    @app.middleware("http")
    async def ip_rate_limit_middleware(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in IP_EXEMPT_PATHS:
            return await call_next(request)
        ip = client_ip(request)
        if not limiter.hit(f"ip:{ip}", *IP_LIMIT):
            logger.info("Rate limit → IP %s over %d/min", ip, IP_LIMIT[0])
            return error_response(ErrorCode.rate_limited)
        return await call_next(request)
