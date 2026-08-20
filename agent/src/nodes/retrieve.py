"""
src/nodes/retrieve.py
----------------------
Two retrieval functions:
1. retrieve_context  — semantic search for specific questions (existing)
2. retrieve_all      — fetch ALL chunks for overview/summary questions (new)

Maps to ERD:
- organization_id → organizations.id (tenant isolation)
- chunk content   → document_chunks.content
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest

from agent.src.models.state import AgentState
from agent.src.models.embeddings import embed_text
from agent.src.utils.spell_correct import correct_query
from agent.src.utils.logger import logger

load_dotenv()

QDRANT_URL      = os.environ.get("QDRANT_URL")
QDRANT_API_KEY  = os.environ.get("QDRANT_API_KEY")
COLLECTION      = "matz_chunks"
TOP_K           = 3
ORGANIZATION_ID = "matz-demo-org"
MIN_SCORE       = 0.1

_qdrant_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def retrieve_context(state: AgentState) -> AgentState:
    """
    Semantic search — finds top K most similar chunks for specific questions.
    Used for all knowledge queries except overview.
    """
    query  = correct_query(state["user_query"])
    state  = {**state, "user_query": query}
    client = _get_client()

    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=ORGANIZATION_ID),
                )
            ]
        ),
        limit=TOP_K,
        with_payload=True,
    ).points

    docs = []
    for hit in results:
        payload = hit.payload
        score   = round(hit.score, 3)

        if score < MIN_SCORE:
            continue

        formatted = (
            f"[Source: {payload['document_title']} · "
            f"{payload['collection_name']} · "
            f"Page {payload['page_number']} | "
            f"Relevance: {score}]\n"
            f"{payload['content']}"
        )
        docs.append(formatted)

    return {**state, "retrieved_docs": docs}


def retrieve_all(state: AgentState) -> AgentState:
    """
    Fetches ALL chunks for this organization from Qdrant.
    Used for overview/summary questions like:
    - "what are all the policies?"
    - "what can you help me with?"
    - "give me a summary of everything"

    When new documents are uploaded and embedded, they are automatically
    included in the response — no code changes needed.
    """
    client = _get_client()
    logger.info("Retrieve all → fetching all chunks for org: %s", ORGANIZATION_ID)

    # Scroll through ALL points for this organization
    all_docs = []
    offset   = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="organization_id",
                        match=MatchValue(value=ORGANIZATION_ID),
                    )
                ]
            ),
            limit=100,          # fetch 100 at a time
            offset=offset,
            with_payload=True,
            with_vectors=False, # no vectors needed for overview
        )

        for point in results:
            payload = point.payload
            formatted = (
                f"[Source: {payload['document_title']} · "
                f"{payload['collection_name']} · "
                f"Page {payload['page_number']}]\n"
                f"{payload['content']}"
            )
            all_docs.append(formatted)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("Retrieve all → fetched %d chunks", len(all_docs))
    return {**state, "retrieved_docs": all_docs, "graded_docs": all_docs}