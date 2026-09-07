"""
data/pipeline/ingestor.py
--------------------------
Ingestor — embeds chunks and uploads to Qdrant Cloud.

This is the final step in the pipeline:
  text chunks → embeddings → Qdrant

Maps to ERD:
  - document_chunks.vector_id     → Qdrant point ID
  - document_chunks.content       → stored in Qdrant payload
  - document_chunks.chunk_index   → stored in Qdrant payload
  - document_chunks.page_number   → stored in Qdrant payload
  - documents.organization_id     → stored in Qdrant payload (tenant isolation)
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from agents.langgraph_agent.embeddings.embeddings import embed_texts
from agents.langgraph_agent.utils.utils import logger

load_dotenv()

QDRANT_URL     = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION     = "matz_chunks"

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def ingest_chunks(
    chunks: list[dict],
    document_id: str,
    document_title: str,
    collection_id: str,
    collection_name: str,
    organization_id: str,
) -> list[str]:
    """
    Embed all chunks and upload to Qdrant.

    Args:
        chunks:          list of chunk dicts from chunker.py
        document_id:     documents.id in ERD
        document_title:  documents.title in ERD
        collection_id:   knowledge_collections.id in ERD
        collection_name: knowledge_collections.name in ERD
        organization_id: organizations.id in ERD (tenant isolation)

    Returns:
        list of Qdrant point IDs (vector_ids) to store in document_chunks.vector_id
    """
    if not chunks:
        logger.warning("Ingestor → no chunks to ingest")
        return []

    client = _get_client()

    # Embed all chunks in one batch — faster than one by one
    logger.info("Ingestor → embedding %d chunks...", len(chunks))
    contents = [chunk["content"] for chunk in chunks]
    vectors  = embed_texts(contents)
    logger.info("Ingestor → embeddings done")

    # Build Qdrant points
    points    = []
    vector_ids = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid4())
        vector_ids.append(point_id)

        points.append(PointStruct(
            id      = point_id,
            vector  = vector,
            payload = {
                # ERD fields
                "chunk_id":        f"chunk-{document_id}-{chunk['chunk_index']}",
                "document_id":     document_id,
                "document_title":  document_title,
                "collection_id":   collection_id,
                "collection_name": collection_name,
                "page_number":     chunk["page_number"],
                "chunk_index":     chunk["chunk_index"],
                "content":         chunk["content"],
                # Tenant isolation — filters in retrieve.py use this
                "organization_id": organization_id,
            }
        ))

    # Upload to Qdrant in batches of 50
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        logger.info(
            "Ingestor → uploaded batch %d/%d (%d points)",
            i // batch_size + 1,
            (len(points) - 1) // batch_size + 1,
            len(batch)
        )

    logger.info(
        "Ingestor → ✅ %d chunks uploaded to Qdrant | doc: %s | org: %s",
        len(points), document_title, organization_id
    )
    return vector_ids


def delete_document_chunks(document_id: str, organization_id: str) -> int:
    """
    Delete all Qdrant points for a document.
    Called when a document is deleted or re-processed.

    Returns number of points deleted.
    """
    client = _get_client()
    result = client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(key="document_id",    match=MatchValue(value=document_id)),
                FieldCondition(key="organization_id", match=MatchValue(value=organization_id)),
            ]
        )
    )
    logger.info("Ingestor → deleted chunks for document %s", document_id)
    return result.operation_id