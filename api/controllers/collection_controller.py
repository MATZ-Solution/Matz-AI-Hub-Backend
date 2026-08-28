"""
api/controllers/collection_controller.py
--------------------------------------------
Backs the Collections page: list/create/delete, merged with live
document counts from Qdrant.
"""

from fastapi import HTTPException

from agent.src.utils.logger import logger
from agent.src.utils.supabase_client import (
    get_collections_from_db, create_collection_in_db, delete_collection_from_db,
)
from api.helpers.qdrant_helper import scroll_all_points
from api.schemas.schemas import CollectionCreate


def get_collections_ctrl(organization_id: str) -> dict:
    """Returns collections from Supabase merged with real document counts from Qdrant."""
    try:
        db_collections = get_collections_from_db(organization_id)

        all_points = scroll_all_points(organization_id)
        col_docs: dict = {}
        for point in all_points:
            col_name = point.payload.get("collection_name", "General")
            doc_id = point.payload.get("document_id", "unknown")
            col_docs.setdefault(col_name, set()).add(doc_id)

        result = []
        for col in db_collections:
            result.append({
                "id": col["id"],
                "name": col["name"],
                "description": col["description"],
                "icon": col["icon"],
                "document_count": len(col_docs.get(col["name"], set())),
                "created_at": col["created_at"],
            })

        # Also include any collections in Qdrant not (yet) in Supabase
        db_names = {c["name"] for c in db_collections}
        for col_name, doc_ids in col_docs.items():
            if col_name not in db_names:
                result.append({
                    "id": None,
                    "name": col_name,
                    "description": "Company knowledge and documents.",
                    "icon": "folder",
                    "document_count": len(doc_ids),
                    "created_at": None,
                })

        logger.info("API → /collections | org: %s | found %d collections", organization_id, len(result))
        return {"collections": result, "total": len(result), "organization_id": organization_id}
    except Exception as e:
        logger.error("Collections error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


def create_collection_ctrl(request: CollectionCreate, organization_id: str) -> dict:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")

    result = create_collection_in_db(
        name=request.name,
        description=request.description,
        icon=request.icon,
        organization_id=organization_id,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create collection")

    logger.info("API → collection created: %s", request.name)
    return {"success": True, "collection": result}


def delete_collection_ctrl(collection_id: str, organization_id: str) -> dict:
    success = delete_collection_from_db(collection_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete collection")
    return {"success": True, "collection_id": collection_id}