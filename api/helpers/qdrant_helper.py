"""
api/helpers/qdrant_helper.py
------------------------------
Shared Qdrant plumbing used by multiple controllers: client creation,
paginated scrolling, and startup index setup.
"""

from api.config.settings import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    QDRANT_TIMEOUT, QDRANT_INDEXED_FIELDS,
)
from agent.src.utils.logger import logger


def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)


def scroll_all_points(organization_id: str) -> list:
    """Paginates through every Qdrant point for an organization."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = get_qdrant_client()
    all_points, offset = [], None
    while True:
        results, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="organization_id", match=MatchValue(value=organization_id))]),
            limit=100, offset=offset, with_payload=True, with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def ensure_qdrant_indexes():
    """Creates payload indexes needed for filtering. Called once on startup."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PayloadSchemaType
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
        for field in QDRANT_INDEXED_FIELDS:
            try:
                client.create_payload_index(collection_name=QDRANT_COLLECTION_NAME, field_name=field, field_schema=PayloadSchemaType.KEYWORD)
                logger.info("Qdrant index ensured: %s", field)
            except Exception:
                pass
        logger.info("Qdrant indexes ready")
    except Exception as e:
        logger.warning("Could not ensure Qdrant indexes: %s", e)
