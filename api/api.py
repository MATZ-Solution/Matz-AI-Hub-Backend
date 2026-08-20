"""
api.py
-------
FastAPI wrapper for the MATZ AI Knowledge Assistant.

Endpoints:
    POST   /chat                        — main chat endpoint
    POST   /ingest                      — document processing pipeline
    GET    /documents                   — list all documents from Qdrant
    DELETE /documents/{id}              — delete document from Qdrant
    PUT    /documents/{id}              — re-process document
    GET    /collections                 — list collections (Qdrant + Supabase merged)
    POST   /collections                 — create new collection in Supabase
    DELETE /collections/{id}            — delete collection from Supabase
    GET    /stats                       — dashboard stats from Qdrant
    POST   /search                      — semantic content search
    POST   /sessions                    — create chat session
    GET    /sessions                    — list all sessions
    DELETE /sessions/{id}               — delete a session
    GET    /sessions/{id}/messages      — get messages for a session
    GET    /health                      — health check
"""

import os
import time
import shutil
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.src.agent.graph import chat
from agent.src.utils.logger import logger
from agent.src.utils.supabase_client import (
    create_session, get_sessions, delete_session,
    save_message, get_messages,
    get_collections_from_db, create_collection_in_db, delete_collection_from_db,
)
from agent.data.pipeline.processor import extract_text
from agent.data.pipeline.chunker import chunk_text
from agent.data.pipeline.ingestor import ingest_chunks


# ── Startup ───────────────────────────────────────────────────────────────────

def ensure_qdrant_indexes():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PayloadSchemaType
        client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"), timeout=30)
        for field in ["organization_id", "document_id", "collection_id"]:
            try:
                client.create_payload_index(collection_name="matz_chunks", field_name=field, field_schema=PayloadSchemaType.KEYWORD)
                logger.info("Qdrant index ensured: %s", field)
            except Exception:
                pass
        logger.info("Qdrant indexes ready")
    except Exception as e:
        logger.warning("Could not ensure Qdrant indexes: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MATZ API starting up...")
    ensure_qdrant_indexes()
    yield
    logger.info("MATZ API shutting down...")


app = FastAPI(title="MATZ AI Knowledge Assistant", description="LangGraph-powered company knowledge assistant API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Qdrant helper ─────────────────────────────────────────────────────────────

def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"), timeout=30)


def scroll_all_points(organization_id: str) -> list:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = get_qdrant_client()
    all_points, offset = [], None
    while True:
        results, next_offset = client.scroll(
            collection_name="matz_chunks",
            scroll_filter=Filter(must=[FieldCondition(key="organization_id", match=MatchValue(value=organization_id))]),
            limit=100, offset=offset, with_payload=True, with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_query: str
    organization_id: str = "matz-demo-org"
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    chat_history: list = []

class CitationResponse(BaseModel):
    chunk_id: Optional[str]
    document_id: Optional[str]
    document_title: str
    collection_name: str
    page_number: int
    cited_text: Optional[str]
    relevance_score: float
    citation_order: int

class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    conversation_id: Optional[str]
    session_id: Optional[str]
    response_time_ms: int
    organization_id: str

class IngestResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    page_count: int
    extraction_method: str
    vector_ids: list[str]

class DocumentItem(BaseModel):
    document_id: str
    document_title: str
    collection_name: str
    page_count: int
    chunk_count: int

class DocumentsResponse(BaseModel):
    documents: list[DocumentItem]
    total: int
    organization_id: str

class CollectionCreate(BaseModel):
    name: str
    description: str = ""
    icon: str = "folder"
    organization_id: str = "matz-demo-org"

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_collections: int
    collections_breakdown: list[dict]
    organization_id: str

class SearchRequest(BaseModel):
    query: str
    organization_id: str = "matz-demo-org"
    collection_name: Optional[str] = None
    top_k: int = 5

class SearchResult(BaseModel):
    document_id: str
    document_title: str
    collection_name: str
    page_number: int
    chunk_content: str
    relevance_score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "MATZ AI Knowledge Assistant API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MATZ AI Knowledge Assistant", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="user_query cannot be empty")

    logger.info("API → /chat | org: %s | query: %s", request.organization_id, request.user_query[:50])
    start_time = time.time()

    try:
        answer, citations, _ = chat(user_query=request.user_query, chat_history=request.chat_history)
    except Exception as e:
        logger.error("Agent error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    response_time_ms = int((time.time() - start_time) * 1000)

    if request.session_id:
        try:
            save_message(request.session_id, "user", request.user_query, [])
            save_message(request.session_id, "assistant", answer, citations)
        except Exception as e:
            logger.warning("Supabase → failed to save messages: %s", e)

    logger.info("API → /chat done | %dms | %d citations", response_time_ms, len(citations))
    return ChatResponse(
        answer=answer,
        citations=[CitationResponse(**c) for c in citations],
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        response_time_ms=response_time_ms,
        organization_id=request.organization_id,
    )


@app.post("/sessions")
async def create_chat_session(organization_id: str = "matz-demo-org"):
    session_id = create_session(organization_id)
    if not session_id:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return {"session_id": session_id}


@app.get("/sessions")
async def list_sessions(organization_id: str = "matz-demo-org"):
    # Only return sessions that have at least one message
    all_sessions = get_sessions(organization_id)
    return {"sessions": all_sessions, "total": len(all_sessions)}


@app.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session")
    return {"success": True, "session_id": session_id}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    messages = get_messages(session_id)
    return {"messages": messages, "session_id": session_id}


@app.get("/collections")
async def get_collections(organization_id: str = "matz-demo-org"):
    """
    Returns collections from Supabase merged with real document counts from Qdrant.
    """
    try:
        # Get collections from Supabase
        db_collections = get_collections_from_db(organization_id)

        # Get document counts from Qdrant
        all_points = scroll_all_points(organization_id)
        col_docs: dict = {}
        for point in all_points:
            col_name = point.payload.get("collection_name", "General")
            doc_id   = point.payload.get("document_id", "unknown")
            if col_name not in col_docs:
                col_docs[col_name] = set()
            col_docs[col_name].add(doc_id)

        # Merge — add document count to each collection
        result = []
        for col in db_collections:
            result.append({
                "id":             col["id"],
                "name":           col["name"],
                "description":    col["description"],
                "icon":           col["icon"],
                "document_count": len(col_docs.get(col["name"], set())),
                "created_at":     col["created_at"],
            })

        # Also include any collections in Qdrant not in Supabase
        db_names = {c["name"] for c in db_collections}
        for col_name, doc_ids in col_docs.items():
            if col_name not in db_names:
                result.append({
                    "id":             None,
                    "name":           col_name,
                    "description":    "Company knowledge and documents.",
                    "icon":           "folder",
                    "document_count": len(doc_ids),
                    "created_at":     None,
                })

        logger.info("API → /collections | org: %s | found %d collections", organization_id, len(result))
        return {"collections": result, "total": len(result), "organization_id": organization_id}

    except Exception as e:
        logger.error("Collections error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
async def create_new_collection(request: CollectionCreate):
    """Creates a new collection in Supabase."""
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")

    result = create_collection_in_db(
        name=request.name,
        description=request.description,
        icon=request.icon,
        organization_id=request.organization_id,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create collection")

    logger.info("API → collection created: %s", request.name)
    return {"success": True, "collection": result}


@app.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Deletes a collection from Supabase."""
    success = delete_collection_from_db(collection_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete collection")
    return {"success": True, "collection_id": collection_id}


@app.get("/documents", response_model=DocumentsResponse)
async def get_documents(organization_id: str = "matz-demo-org"):
    try:
        all_points = scroll_all_points(organization_id)
        docs_map: dict = {}
        for point in all_points:
            payload = point.payload
            doc_id  = payload.get("document_id", "unknown")
            if doc_id not in docs_map:
                docs_map[doc_id] = {"document_id": doc_id, "document_title": payload.get("document_title", "Unknown"), "collection_name": payload.get("collection_name", "General"), "page_count": payload.get("page_number", 1), "chunk_count": 1}
            else:
                docs_map[doc_id]["page_count"] = max(docs_map[doc_id]["page_count"], payload.get("page_number", 1))
                docs_map[doc_id]["chunk_count"] += 1
        documents = list(docs_map.values())
        logger.info("API → /documents | org: %s | found %d documents", organization_id, len(documents))
        return DocumentsResponse(documents=[DocumentItem(**d) for d in documents], total=len(documents), organization_id=organization_id)
    except Exception as e:
        logger.error("Documents error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, organization_id: str = "matz-demo-org"):
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_qdrant_client()
        client.delete(collection_name="matz_chunks", points_selector=Filter(must=[
            FieldCondition(key="document_id",     match=MatchValue(value=document_id)),
            FieldCondition(key="organization_id", match=MatchValue(value=organization_id)),
        ]))
        logger.info("API → deleted document %s", document_id)
        return {"success": True, "document_id": document_id}
    except Exception as e:
        logger.error("Delete error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/documents/{document_id}")
async def update_document(
    document_id: str,
    file: UploadFile = File(...),
    document_title: str = Form(...),
    collection_name: str = Form(...),
    collection_id: str = Form(...),
    organization_id: str = Form(default="matz-demo-org"),
):
    logger.info("API → /documents/%s update", document_id)
    allowed_extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".txt"}
    file_ext = "." + file.filename.split(".")[-1].lower() if file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    tmp_path = None
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_qdrant_client()
        client.delete(collection_name="matz_chunks", points_selector=Filter(must=[
            FieldCondition(key="document_id",     match=MatchValue(value=document_id)),
            FieldCondition(key="organization_id", match=MatchValue(value=organization_id)),
        ]))
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = extract_text(tmp_path)
        if not result["success"]:
            raise HTTPException(status_code=422, detail=f"Text extraction failed: {result['error']}")
        chunks = chunk_text(text=result["text"], document_id=document_id, page_count=result["page_count"])
        if not chunks:
            raise HTTPException(status_code=422, detail="No content could be extracted")
        vector_ids = ingest_chunks(chunks=chunks, document_id=document_id, document_title=document_title, collection_id=collection_id, collection_name=collection_name, organization_id=organization_id)
        return {"success": True, "document_id": document_id, "chunks_created": len(chunks), "page_count": result["page_count"], "extraction_method": result["method"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Update error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_title: str = Form(...),
    collection_name: str = Form(...),
    collection_id: str = Form(...),
    organization_id: str = Form(default="matz-demo-org"),
    document_id: str = Form(...),
):
    logger.info("API → /ingest | doc: %s | org: %s", document_title, organization_id)
    allowed_extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".txt"}
    file_ext = "." + file.filename.split(".")[-1].lower() if file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
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
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/stats", response_model=StatsResponse)
async def get_stats(organization_id: str = "matz-demo-org"):
    try:
        all_points = scroll_all_points(organization_id)
        doc_ids: set = set()
        col_docs: dict = {}
        for point in all_points:
            payload  = point.payload
            doc_ids.add(payload.get("document_id", "unknown"))
            col_name = payload.get("collection_name", "General")
            if col_name not in col_docs:
                col_docs[col_name] = set()
            col_docs[col_name].add(payload.get("document_id", "unknown"))

        # Collection count comes from Supabase (source of truth for collections),
        # not from Qdrant — a collection created in Supabase with no documents
        # yet ingested should still be counted.
        db_collections = get_collections_from_db(organization_id)
        total_collections = len(db_collections)

        # Build breakdown from every Supabase collection, filling in document
        # counts from Qdrant where available (0 for collections with no docs yet).
        collections_breakdown = [
            {"collection": col["name"], "document_count": len(col_docs.get(col["name"], set()))}
            for col in db_collections
        ]
        # Include any Qdrant collections not (yet) present in Supabase, so document
        # counts are never silently dropped.
        db_names = {col["name"] for col in db_collections}
        for col_name, docs in col_docs.items():
            if col_name not in db_names:
                collections_breakdown.append({"collection": col_name, "document_count": len(docs)})

        logger.info("API → /stats | org: %s | docs: %d | chunks: %d | collections: %d", organization_id, len(doc_ids), len(all_points), total_collections)
        return StatsResponse(total_documents=len(doc_ids), total_chunks=len(all_points), total_collections=total_collections, collections_breakdown=collections_breakdown, organization_id=organization_id)
    except Exception as e:
        logger.error("Stats error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_content(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from agent.src.models.embeddings import embed_text
        client = get_qdrant_client()
        must_conditions = [FieldCondition(key="organization_id", match=MatchValue(value=request.organization_id))]
        if request.collection_name:
            must_conditions.append(FieldCondition(key="collection_name", match=MatchValue(value=request.collection_name)))
        query_vector = embed_text(request.query)
        results = client.query_points(collection_name="matz_chunks", query=query_vector, query_filter=Filter(must=must_conditions), limit=request.top_k, with_payload=True).points
        search_results = []
        for hit in results:
            payload = hit.payload
            score   = round(hit.score, 3)
            if score < 0.2:
                continue
            content = payload.get("content", "")
            excerpt = content[:200] + "..." if len(content) > 200 else content
            search_results.append(SearchResult(document_id=payload.get("document_id", ""), document_title=payload.get("document_title", ""), collection_name=payload.get("collection_name", ""), page_number=payload.get("page_number", 1), chunk_content=excerpt, relevance_score=score))
        logger.info("API → /search | query: '%s' | found %d results", request.query[:30], len(search_results))
        return SearchResponse(results=search_results, query=request.query, total=len(search_results))
    except Exception as e:
        logger.error("Search error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))