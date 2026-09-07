"""
api/controllers/knowledge_controller.py
-------------------------------------------
Backs the Knowledge page: documents list/delete/update, ingestion
pipeline, and semantic search.
"""

from fastapi import HTTPException, UploadFile

from agents.langgraph_agent.utils.utils import logger
from data.pipeline.processor import extract_text
from data.pipeline.chunker import chunk_text
from data.pipeline.ingestor import ingest_chunks
from api.helpers.qdrant_helper import get_qdrant_client, scroll_all_points
from api.helpers.file_helper import is_allowed_extension, save_upload_to_tempfile, cleanup_tempfile
from api.config.settings import QDRANT_COLLECTION_NAME, SEARCH_MIN_RELEVANCE_SCORE, SEARCH_EXCERPT_LENGTH
from api.schemas.schemas import (
    DocumentsResponse, DocumentItem, IngestResponse,
    SearchRequest, SearchResponse, SearchResult,
)


def get_documents_ctrl(organization_id: str) -> DocumentsResponse:
    try:
        all_points = scroll_all_points(organization_id)
        docs_map: dict = {}
        for point in all_points:
            payload = point.payload
            doc_id = payload.get("document_id", "unknown")
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "document_title": payload.get("document_title", "Unknown"),
                    "collection_name": payload.get("collection_name", "General"),
                    "page_count": payload.get("page_number", 1),
                    "chunk_count": 1,
                }
            else:
                docs_map[doc_id]["page_count"] = max(docs_map[doc_id]["page_count"], payload.get("page_number", 1))
                docs_map[doc_id]["chunk_count"] += 1
        documents = list(docs_map.values())
        logger.info("API → /documents | org: %s | found %d documents", organization_id, len(documents))
        return DocumentsResponse(documents=[DocumentItem(**d) for d in documents], total=len(documents), organization_id=organization_id)
    except Exception as e:
        logger.error("Documents error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


def delete_document_ctrl(document_id: str, organization_id: str) -> dict:
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_qdrant_client()
        client.delete(collection_name=QDRANT_COLLECTION_NAME, points_selector=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="organization_id", match=MatchValue(value=organization_id)),
        ]))
        logger.info("API → deleted document %s", document_id)
        return {"success": True, "document_id": document_id}
    except Exception as e:
        logger.error("Delete error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


def update_document_ctrl(
    document_id: str,
    file: UploadFile,
    document_title: str,
    collection_name: str,
    collection_id: str,
    organization_id: str,
) -> dict:
    logger.info("API → /documents/%s update", document_id)
    if not is_allowed_extension(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    tmp_path = None
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_qdrant_client()
        client.delete(collection_name=QDRANT_COLLECTION_NAME, points_selector=Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            FieldCondition(key="organization_id", match=MatchValue(value=organization_id)),
        ]))
        tmp_path = save_upload_to_tempfile(file)
        result = extract_text(tmp_path)
        if not result["success"]:
            raise HTTPException(status_code=422, detail=f"Text extraction failed: {result['error']}")
        chunks = chunk_text(text=result["text"], document_id=document_id, page_count=result["page_count"])
        if not chunks:
            raise HTTPException(status_code=422, detail="No content could be extracted")
        ingest_chunks(chunks=chunks, document_id=document_id, document_title=document_title, collection_id=collection_id, collection_name=collection_name, organization_id=organization_id)
        return {"success": True, "document_id": document_id, "chunks_created": len(chunks), "page_count": result["page_count"], "extraction_method": result["method"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Update error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")
    finally:
        cleanup_tempfile(tmp_path)


def ingest_document_ctrl(
    file: UploadFile,
    document_title: str,
    collection_name: str,
    collection_id: str,
    organization_id: str,
    document_id: str,
) -> IngestResponse:
    logger.info("API → /ingest | doc: %s | org: %s", document_title, organization_id)
    if not is_allowed_extension(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    tmp_path = None
    try:
        tmp_path = save_upload_to_tempfile(file)
        result = extract_text(tmp_path)
        if not result["success"]:
            raise HTTPException(status_code=422, detail=f"Text extraction failed: {result['error']}")
        chunks = chunk_text(text=result["text"], document_id=document_id, page_count=result["page_count"])
        if not chunks:
            raise HTTPException(status_code=422, detail="No content could be extracted")
        vector_ids = ingest_chunks(chunks=chunks, document_id=document_id, document_title=document_title, collection_id=collection_id, collection_name=collection_name, organization_id=organization_id)
        logger.info("API → /ingest done | %d chunks | doc: %s", len(vector_ids), document_title)
        return IngestResponse(success=True, document_id=document_id, chunks_created=len(chunks), page_count=result["page_count"], extraction_method=result["method"], vector_ids=vector_ids)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingest error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        cleanup_tempfile(tmp_path)


def search_content_ctrl(request: SearchRequest, organization_id: str) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from agents.langgraph_agent.embeddings.embeddings import embed_text
        client = get_qdrant_client()
        must_conditions = [FieldCondition(key="organization_id", match=MatchValue(value=organization_id))]
        if request.collection_name:
            must_conditions.append(FieldCondition(key="collection_name", match=MatchValue(value=request.collection_name)))
        query_vector = embed_text(request.query)
        results = client.query_points(collection_name=QDRANT_COLLECTION_NAME, query=query_vector, query_filter=Filter(must=must_conditions), limit=request.top_k, with_payload=True).points
        search_results = []
        for hit in results:
            payload = hit.payload
            score = round(hit.score, 3)
            if score < SEARCH_MIN_RELEVANCE_SCORE:
                continue
            content = payload.get("content", "")
            excerpt = content[:SEARCH_EXCERPT_LENGTH] + "..." if len(content) > SEARCH_EXCERPT_LENGTH else content
            search_results.append(SearchResult(document_id=payload.get("document_id", ""), document_title=payload.get("document_title", ""), collection_name=payload.get("collection_name", ""), page_number=payload.get("page_number", 1), chunk_content=excerpt, relevance_score=score))
        logger.info("API → /search | query: '%s' | found %d results", request.query[:30], len(search_results))
        return SearchResponse(results=search_results, query=request.query, total=len(search_results))
    except Exception as e:
        logger.error("Search error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))