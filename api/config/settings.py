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

REQUIRED_ENV_VARS = ("GROQ_API_KEY", "QDRANT_URL", "SUPABASE_URL", "SUPABASE_KEY")


def missing_required_env() -> list[str]:
    """Names of required environment variables that are unset or blank."""
    return [name for name in REQUIRED_ENV_VARS if not (os.environ.get(name) or "").strip()]


# ── Health check ──────────────────────────────────────────────────────────────
# /health makes a real 1-token provider call. Fly probes it every 30 s, so the
# provider result is cached: ~288 provider calls a day instead of ~2,900.
HEALTH_PROVIDER_CACHE_SECONDS = 300

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "matz_chunks"
QDRANT_TIMEOUT = 30
QDRANT_INDEXED_FIELDS = ["organization_id", "document_id", "collection_id"]

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
