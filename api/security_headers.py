"""
api/security_headers.py
------------------------
Hardening headers on every API response (task A6). The API serves JSON only,
so its CSP allows nothing at all and it may never be framed.
"""

from fastapi import Request

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "X-Robots-Tag": "noindex, nofollow",
    "Cache-Control": "no-store",
}


def install_security_headers(app) -> None:
    """Install last, so it is the outermost layer and also covers CORS
    preflights and error responses."""

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response
