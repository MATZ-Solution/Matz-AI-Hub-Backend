"""
src/utils/config.py
--------------------
Single place to read and validate all environment variables.
Import `cfg` anywhere instead of sprinkling os.environ.get() calls
across the codebase.

Usage:
    from src.utils.config import cfg
    print(cfg.qdrant_url)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Qdrant
    qdrant_url: str     = os.environ.get("QDRANT_URL", "")
    qdrant_api_key: str = os.environ.get("QDRANT_API_KEY", "")

    # Groq
    groq_api_key: str   = os.environ.get("GROQ_API_KEY", "")

    # Agent settings
    collection_name: str    = "matz_chunks"
    embed_model: str        = "all-MiniLM-L6-v2"
    llm_model: str          = "openai/gpt-oss-120b"
    top_k: int              = 3
    min_score: float        = 0.2
    organization_id: str    = "matz-demo-org"
    max_history_messages: int = 20


# Singleton — import this directly
cfg = Config()