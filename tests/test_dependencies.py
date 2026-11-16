"""
tests/test_dependencies.py — the real service clients must import.

The rest of the suite replaces Supabase and Qdrant with fakes, so a missing
transitive dependency (PyJWT, needed by supabase-auth) once went unnoticed
while every real Supabase call, sign-in included, failed with a 500.
"""


def test_supabase_client_imports():
    from supabase import create_client  # noqa: F401  (imports jwt)


def test_qdrant_client_imports():
    from qdrant_client import QdrantClient  # noqa: F401
