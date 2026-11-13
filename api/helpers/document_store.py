"""
api/helpers/document_store.py
------------------------------
Document-level rows in Postgres (migration 004_documents.sql). Chunks live in
Qdrant; this table records when each document was added, replaced or
deleted, which the Overview activity chart (B1) counts.

Writes are best effort, like chat_events: a failure is logged and never
fails the upload or delete the user asked for.
"""

from datetime import datetime, timezone

from agents.langgraph_agent.utils import utils as db
from agents.langgraph_agent.utils.utils import logger, _exec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_added(organization_id: str, document_id: str, title: str,
                 collection_id: str, collection_name: str, user_id: str | None) -> None:
    now = _now()
    try:
        _exec(db.get_supabase().table("documents").upsert({
            "organization_id": organization_id, "id": document_id, "title": title,
            "collection_id": collection_id, "collection_name": collection_name,
            "version": 1, "created_by": user_id, "created_at": now,
            "updated_by": user_id, "updated_at": now, "deleted_at": None,
        }, on_conflict="organization_id,id"))
    except Exception as e:
        logger.error("Supabase → documents insert failed (is migration 004 applied?): %s", e)


def record_replaced(organization_id: str, document_id: str, title: str,
                    collection_id: str, collection_name: str, user_id: str | None) -> None:
    try:
        client = db.get_supabase()
        rows = _exec(client.table("documents").select("version")
                     .eq("organization_id", organization_id).eq("id", document_id).limit(1)).data or []
        if not rows:
            # A document uploaded before this table existed: start its history now.
            record_added(organization_id, document_id, title, collection_id, collection_name, user_id)
            return
        _exec(client.table("documents").update({
            "title": title, "collection_id": collection_id, "collection_name": collection_name,
            "version": (rows[0].get("version") or 1) + 1,
            "updated_by": user_id, "updated_at": _now(),
        }).eq("organization_id", organization_id).eq("id", document_id))
    except Exception as e:
        logger.error("Supabase → documents update failed: %s", e)


def record_deleted(organization_id: str, document_id: str) -> None:
    try:
        _exec(db.get_supabase().table("documents").update({"deleted_at": _now()})
              .eq("organization_id", organization_id).eq("id", document_id))
    except Exception as e:
        logger.error("Supabase → documents soft delete failed: %s", e)
