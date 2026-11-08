"""
api/config/settings.py
-----------------------
Central place for environment variables and app-wide constants.
Nothing else in api/ should read os.environ directly — import from here instead.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Required secrets ──────────────────────────────────────────────────────────
# Secrets come from the environment only (Fly/Render secrets, or a local .env
# that git ignores) — never from code. The API refuses to start without them;
# see missing_required_env() and api/app.py.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_FAST = os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

REQUIRED_ENV_VARS = ("GROQ_API_KEY", "QDRANT_URL", "SUPABASE_URL", "SUPABASE_KEY", "AUTH_SECRET")

# Interactive docs (/docs, /redoc, /openapi.json) are off unless explicitly
# enabled, because every route except /health requires a token (A3).
ENABLE_API_DOCS = os.environ.get("ENABLE_API_DOCS", "").lower() in ("1", "true", "yes")


def missing_required_env() -> list[str]:
    """Names of required environment variables that are unset or blank."""
    return [name for name in REQUIRED_ENV_VARS if not (os.environ.get(name) or "").strip()]


# ── Health check ──────────────────────────────────────────────────────────────
# /health makes a real 1-token provider call. Fly probes it every 30 s, so the
# provider result is cached: ~288 provider calls a day instead of ~2,900.
HEALTH_PROVIDER_CACHE_SECONDS = 300

# ── Qdrant ────────────────────────────────────────────────────────────────────
# Connection and collection settings live in data/qdrant_store.py, the only
# module allowed to talk to Qdrant (task A4).

# ── App defaults ──────────────────────────────────────────────────────────────
DEFAULT_ORGANIZATION_ID = "matz-demo-org"

# ── Uploads ───────────────────────────────────────────────────────────────────
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt"}

# ── Search ────────────────────────────────────────────────────────────────────
SEARCH_MIN_RELEVANCE_SCORE = 0.2
SEARCH_EXCERPT_LENGTH = 200

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# ── App metadata ──────────────────────────────────────────────────────────────
APP_TITLE = "MATZ AI Knowledge Assistant"
APP_DESCRIPTION = "LangGraph-powered company knowledge assistant API"
APP_VERSION = "1.0.0"
