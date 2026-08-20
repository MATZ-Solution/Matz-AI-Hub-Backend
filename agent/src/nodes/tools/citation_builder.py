"""
src/nodes/tools/citation_builder.py
-------------------------------------
Citation Builder node — the last node in the MATZ agent graph.

What it does:
1. Takes the graded chunks used to generate the answer
2. Extracts structured citation data from each chunk's payload
3. Builds a list of Citation objects ready to write to message_citations table

Maps directly to ERD:
  - chunk_id        → message_citations.chunk_id  → document_chunks.id
  - document_id     → message_citations.document_id → documents.id
  - document_title  → documents.title
  - page_number     → message_citations.page_number
  - relevance_score → message_citations.relevance_score
  - citation_order  → message_citations.citation_order

Note: In MVP we extract metadata from the chunk text itself since
we don't have a live PostgreSQL connection yet. When NestJS backend
is ready, swap _parse_chunk_metadata() with a real DB lookup.
"""

import re
from agent.src.models.state import AgentState
from agent.src.utils.logger import logger


def build_citations(state: AgentState) -> AgentState:
    """
    Builds structured citation list from graded chunks.
    Citations are included in the final API response and
    written to message_citations table by NestJS backend.
    """
    chunks = state.get("graded_docs") or []

    if not chunks:
        logger.info("Citation builder → no chunks, no citations")
        return {**state, "citations": []}

    citations = []
    for order, chunk in enumerate(chunks, start=1):
        meta = _parse_chunk_metadata(chunk)
        if meta:
            citation = {
                # message_citations table fields
                "chunk_id":       meta["chunk_id"],
                "document_id":    meta["document_id"],
                "document_title": meta["document_title"],
                "collection_name":meta["collection_name"],
                "page_number":    meta["page_number"],
                "cited_text":     meta["cited_text"],
                "relevance_score":meta["relevance_score"],
                "citation_order": order,
            }
            citations.append(citation)
            logger.info(
                "Citation builder → [%d] %s · page %d",
                order, meta["document_title"], meta["page_number"]
            )

    logger.info("Citation builder → %d citations built", len(citations))
    return {**state, "citations": citations}


def _parse_chunk_metadata(chunk: str) -> dict | None:
    """
    Extracts metadata from chunk formatted string.

    Chunk format (set in retrieve.py):
    [Source: {document_title} · {collection_name} · Page {page_number} | Relevance: {score}]
    {content}
    """
    try:
        # Extract the source line
        source_match = re.search(
          r'\[Source: (.+?) · (.+?) · Page (\d+)(?:\s*\|\s*Relevance:\s*([\d.]+))?\]',
          chunk
        )
            
            
        
        if not source_match:
            logger.warning("Citation builder → could not parse chunk metadata")
            return None

        document_title  = source_match.group(1).strip()
        collection_name = source_match.group(2).strip()
        page_number     = int(source_match.group(3))
        relevance_score = float(source_match.group(4)) if source_match.group(4) else 1.0

        # Extract content (everything after the source line)
        content_start = chunk.find(']') + 1
        cited_text    = chunk[content_start:].strip()

        # Keep cited_text short — first 200 chars
        if len(cited_text) > 200:
            cited_text = cited_text[:200] + "..."

        # Build chunk_id and document_id from title
        # In production these come from Qdrant payload → PostgreSQL lookup
        slug         = document_title.lower().replace(" ", "-").replace("&", "and")
        chunk_id     = f"chunk-{slug}"
        document_id  = f"doc-{slug}"

        return {
            "chunk_id":       chunk_id,
            "document_id":    document_id,
            "document_title": document_title,
            "collection_name":collection_name,
            "page_number":    page_number,
            "cited_text":     cited_text,
            "relevance_score":relevance_score,
        }

    except Exception as e:
        logger.warning("Citation builder → parse error: %s", e)
        return None