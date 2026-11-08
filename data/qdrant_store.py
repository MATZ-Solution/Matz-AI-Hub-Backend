"""
data/qdrant_store.py
---------------------
The ONLY module that talks to Qdrant (task A4).

Every query, scroll, upsert and delete goes through a function here that
takes an organization id and builds its filter with `org_filter()`, so no
code path can read, change or cite another organization's chunks.
tests/test_tenancy.py fails if a Qdrant call appears anywhere else.

All organizations share one collection; the `organization_id` payload field
is a keyword index marked as the tenant index (is_tenant), which Qdrant uses
to keep each tenant's points together.
"""

import os
import threading

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, KeywordIndexParams, MatchValue, PointStruct,
)

load_dotenv()

COLLECTION = "matz_chunks"
TENANT_FIELD = "organization_id"
TIMEOUT_SECONDS = 30
SCROLL_PAGE = 256

# Payload fields used in filters. The tenant field is created as the tenant
# index; the others are plain keyword indexes.
INDEXED_FIELDS = ("document_id", "collection_id", "collection_name")

_client: QdrantClient | None = None
_client_lock = threading.Lock()


class TenantError(ValueError):
    """A call would cross or ignore the tenant boundary."""


def _get_client() -> QdrantClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = QdrantClient(
                url=os.environ.get("QDRANT_URL"),
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=TIMEOUT_SECONDS,
            )
        return _client


# ── Filters ──────────────────────────────────────────────────────────────────

def match(key: str, value) -> FieldCondition:
    """A `key == value` condition, for the `conditions` of the calls below."""
    if key == TENANT_FIELD:
        raise TenantError("organization_id is set by org_filter(), not passed as a condition")
    return FieldCondition(key=key, match=MatchValue(value=value))


def org_filter(org_id: str, *conditions: FieldCondition) -> Filter:
    """The filter every Qdrant operation uses: this org, plus any conditions."""
    if not isinstance(org_id, str) or not org_id.strip():
        raise TenantError("an organization id is required for every Qdrant operation")
    return Filter(must=[FieldCondition(key=TENANT_FIELD, match=MatchValue(value=org_id)), *conditions])


# ── Reads ────────────────────────────────────────────────────────────────────

def query(org_id: str, vector: list[float], *conditions: FieldCondition, limit: int = 5):
    """Nearest chunks to `vector` within this org. Returns scored points."""
    return _get_client().query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=org_filter(org_id, *conditions),
        limit=limit,
        with_payload=True,
    ).points


def scroll_all(org_id: str, *conditions: FieldCondition) -> list:
    """Every point in this org (payload only, no vectors), paginated."""
    client = _get_client()
    points, offset = [], None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=org_filter(org_id, *conditions),
            limit=SCROLL_PAGE, offset=offset,
            with_payload=True, with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            return points


def exists(org_id: str, *conditions: FieldCondition) -> bool:
    """True if this org has at least one point matching `conditions`."""
    page, _ = _get_client().scroll(
        collection_name=COLLECTION,
        scroll_filter=org_filter(org_id, *conditions),
        limit=1, with_payload=False, with_vectors=False,
    )
    return bool(page)


# ── Writes ───────────────────────────────────────────────────────────────────

def upsert(org_id: str, points: list[PointStruct], batch_size: int = 50) -> None:
    """Writes points that must all belong to `org_id`."""
    org_filter(org_id)  # validates org_id
    for point in points:
        if (point.payload or {}).get(TENANT_FIELD) != org_id:
            raise TenantError("every upserted point must carry this organization's id")
    client = _get_client()
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION, points=points[i:i + batch_size])


def delete(org_id: str, *conditions: FieldCondition) -> None:
    """Deletes this org's points matching `conditions`. At least one condition
    is required, so a missing argument can never wipe a whole organization."""
    if not conditions:
        raise TenantError("delete needs at least one condition besides the organization")
    _get_client().delete(collection_name=COLLECTION, points_selector=org_filter(org_id, *conditions))


# ── Operations ───────────────────────────────────────────────────────────────

def ping() -> None:
    """Raises if Qdrant or the collection is unreachable (used by /health)."""
    _get_client().get_collection(COLLECTION)


def ensure_indexes(logger) -> None:
    """Creates payload indexes on startup; the tenant field as the tenant index."""
    client = _get_client()
    specs = [(TENANT_FIELD, KeywordIndexParams(type="keyword", is_tenant=True))]
    specs += [(field, KeywordIndexParams(type="keyword")) for field in INDEXED_FIELDS]
    for field, schema in specs:
        try:
            client.create_payload_index(collection_name=COLLECTION, field_name=field, field_schema=schema)
            logger.info("Qdrant index ensured: %s", field)
        except Exception as e:
            logger.warning("Qdrant index %s not updated: %s", field, e)
