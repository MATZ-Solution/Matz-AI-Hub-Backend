"""
api/controllers/overview_controller.py
------------------------------------------
Backs the Overview (dashboard) page: /stats.
"""

from fastapi import HTTPException

from agent.src.utils.logger import logger
from agent.src.utils.supabase_client import get_collections_from_db
from api.helpers.qdrant_helper import scroll_all_points
from api.schemas.schemas import StatsResponse
from api.config.settings import DEFAULT_ORGANIZATION_ID


async def get_stats_ctrl(organization_id: str = DEFAULT_ORGANIZATION_ID) -> StatsResponse:
    try:
        all_points = scroll_all_points(organization_id)
        doc_ids: set = set()
        col_docs: dict = {}
        for point in all_points:
            payload = point.payload
            doc_ids.add(payload.get("document_id", "unknown"))
            col_name = payload.get("collection_name", "General")
            col_docs.setdefault(col_name, set()).add(payload.get("document_id", "unknown"))

        # Collection count comes from Supabase (source of truth for collections),
        # not from Qdrant — a collection created in Supabase with no documents
        # yet ingested should still be counted.
        db_collections = get_collections_from_db(organization_id)
        total_collections = len(db_collections)

        collections_breakdown = [
            {"collection": col["name"], "document_count": len(col_docs.get(col["name"], set()))}
            for col in db_collections
        ]
        db_names = {col["name"] for col in db_collections}
        for col_name, docs in col_docs.items():
            if col_name not in db_names:
                collections_breakdown.append({"collection": col_name, "document_count": len(docs)})

        logger.info("API → /stats | org: %s | docs: %d | chunks: %d | collections: %d", organization_id, len(doc_ids), len(all_points), total_collections)
        return StatsResponse(
            total_documents=len(doc_ids),
            total_chunks=len(all_points),
            total_collections=total_collections,
            collections_breakdown=collections_breakdown,
            organization_id=organization_id,
        )
    except Exception as e:
        logger.error("Stats error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
