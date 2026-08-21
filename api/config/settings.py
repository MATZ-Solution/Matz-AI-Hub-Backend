"""
api/config/settings.py
-----------------------
Central place for environment variables and app-wide constants.
Nothing else in api/ should read os.environ directly — import from here instead.
"""

import os
from dotenv import load_dotenv

load_dotenv()

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
