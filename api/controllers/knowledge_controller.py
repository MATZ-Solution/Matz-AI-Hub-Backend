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
from api.helpers.drive_helper import (
    DriveError, extract_folder_id, list_folder_files, download_file, MAX_FILES,
)
from api.config.settings import QDRANT_COLLECTION_NAME, SEARCH_MIN_RELEVANCE_SCORE, SEARCH_EXCERPT_LENGTH
from api.schemas.schemas import (
    DocumentsResponse, DocumentItem, IngestResponse,
    SearchRequest, SearchResponse, SearchResult,
    DriveIngestRequest, DriveIngestResponse, DriveFileResult,
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

def ingest_from_drive_ctrl(
    request: DriveIngestRequest,
    organization_id: str,
) -> DriveIngestResponse:
    """
    Ingest every supported document in a public Google Drive folder.

    Runs synchronously: fine for the folder sizes this is meant for (a handful
    of policy documents), and it keeps the client simple. A folder large enough
    to time out would need a background job with a status endpoint — the
    MAX_FILES cap in drive_helper keeps that from creeping up unnoticed.

    Re-running the same folder is safe. Each file's document_id is derived from
    its Drive file id, and the existing points are deleted before re-ingesting,
    so an import is idempotent rather than duplicating every chunk.
    """
    try:
        folder_id = extract_folder_id(request.folder_url)
        files = list_folder_files(folder_id)
    except DriveError as e:
        # A configuration or permission problem — the message is written to be
        # shown to the user as-is.
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "API → /ingest/drive | folder: %s | %d file(s) | collection: %s",
        folder_id, len(files), request.collection_name,
    )

    if not files:
        return DriveIngestResponse(
            success=True, folder_id=folder_id, files_found=0,
            files_ingested=0, files_failed=0, total_chunks=0, results=[],
        )

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = get_qdrant_client()

    results: list[DriveFileResult] = []
    total_chunks = 0

    for drive_file in files[:MAX_FILES]:
        file_id = drive_file["id"]
        name = drive_file.get("name", file_id)
        # Deterministic, so a re-import updates rather than duplicates.
        document_id = f"gdrive-{file_id}"
        document_title = name.rsplit(".", 1)[0] if "." in name else name

        tmp_path = None
        try:
            tmp_path = download_file(file_id, drive_file.get("mimeType", ""))

            extraction = extract_text(tmp_path)
            if not extraction["success"]:
                results.append(DriveFileResult(
                    file_id=file_id, name=name, status="failed",
                    error=f"Text extraction failed: {extraction['error']}",
                ))
                continue

            chunks = chunk_text(
                text=extraction["text"],
                document_id=document_id,
                page_count=extraction["page_count"],
            )
            if not chunks:
                results.append(DriveFileResult(
                    file_id=file_id, name=name, status="failed",
                    error="No content could be extracted",
                ))
                continue

            # Clear any previous import of this same Drive file first.
            client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=Filter(must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="organization_id", match=MatchValue(value=organization_id)),
                ]),
            )

            ingest_chunks(
                chunks=chunks,
                document_id=document_id,
                document_title=document_title,
                collection_id=request.collection_id,
                collection_name=request.collection_name,
                organization_id=organization_id,
            )

            total_chunks += len(chunks)
            results.append(DriveFileResult(
                file_id=file_id, name=name, status="ingested",
                chunks_created=len(chunks),
                page_count=extraction["page_count"],
                extraction_method=extraction["method"],
            ))
            logger.info("Drive → ingested %s (%d chunks)", name, len(chunks))

        except DriveError as e:
            # One bad file should not abandon the rest of the folder.
            logger.warning("Drive → %s failed: %s", name, e)
            results.append(DriveFileResult(
                file_id=file_id, name=name, status="failed", error=str(e),
            ))
        except Exception as e:
            logger.error("Drive → %s failed: %s", name, e)
            results.append(DriveFileResult(
                file_id=file_id, name=name, status="failed", error=str(e),
            ))
        finally:
            cleanup_tempfile(tmp_path)

    ingested = sum(1 for r in results if r.status == "ingested")
    failed = sum(1 for r in results if r.status == "failed")

    logger.info(
        "API → /ingest/drive done | %d ingested, %d failed | %d chunks",
        ingested, failed, total_chunks,
    )

    return DriveIngestResponse(
        # Successful as long as something landed; per-file errors are in results.
        success=ingested > 0,
        folder_id=folder_id,
        files_found=len(files),
        files_ingested=ingested,
        files_failed=failed,
        total_chunks=total_chunks,
        results=results,
    )
