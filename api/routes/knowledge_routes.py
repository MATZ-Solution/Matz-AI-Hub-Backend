"""api/routes/knowledge_routes.py — Knowledge page (documents, ingest, search)."""

from fastapi import APIRouter, Depends, UploadFile, File, Form

from api.controllers.knowledge_controller import (
    get_documents_ctrl, delete_document_ctrl, update_document_ctrl,
    ingest_document_ctrl, search_content_ctrl, ingest_from_drive_ctrl,
)
from api.schemas.schemas import (
    DocumentsResponse, IngestResponse, SearchRequest, SearchResponse,
    DriveIngestRequest, DriveIngestResponse,
)
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["knowledge"])


@router.get("/documents", response_model=DocumentsResponse)
def get_documents(user: CurrentUser = Depends(get_current_user)):
    return get_documents_ctrl(user.organization_id)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, user: CurrentUser = Depends(get_current_user)):
    return delete_document_ctrl(document_id, user.organization_id)


@router.put("/documents/{document_id}")
def update_document(
    document_id: str,
    file: UploadFile = File(...),
    document_title: str = Form(...),
    collection_name: str = Form(...),
    collection_id: str = Form(...),
    user: CurrentUser = Depends(get_current_user),
):
    return update_document_ctrl(
        document_id, file, document_title, collection_name, collection_id, user.organization_id
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile = File(...),
    document_title: str = Form(...),
    collection_name: str = Form(...),
    collection_id: str = Form(...),
    document_id: str = Form(...),
    user: CurrentUser = Depends(get_current_user),
):
    return ingest_document_ctrl(
        file, document_title, collection_name, collection_id, user.organization_id, document_id
    )


@router.post("/ingest/drive", response_model=DriveIngestResponse)
def ingest_from_drive(
    request: DriveIngestRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Import every supported document from a public Google Drive folder."""
    return ingest_from_drive_ctrl(request, user.organization_id)


@router.post("/search", response_model=SearchResponse)
def search_content(request: SearchRequest, user: CurrentUser = Depends(get_current_user)):
    return search_content_ctrl(request, user.organization_id)