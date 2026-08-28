"""
api/schemas/schemas.py
------------------------
All Pydantic request/response models used across the API, in one place
so controllers and routes can share the same contracts.
"""

from typing import Optional
from pydantic import BaseModel

from api.config.settings import DEFAULT_ORGANIZATION_ID


# ── Assistant (chat / sessions) ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_query: str
    organization_id: str = DEFAULT_ORGANIZATION_ID
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


# ── Knowledge (documents / ingest / search) ──────────────────────────────────

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


class SearchRequest(BaseModel):
    query: str
    organization_id: str = DEFAULT_ORGANIZATION_ID
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


# ── Collections ───────────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str
    description: str = ""
    icon: str = "folder"
    organization_id: str = DEFAULT_ORGANIZATION_ID


# ── Overview ──────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_collections: int
    collections_breakdown: list[dict]
    organization_id: str

# ── Settings: Workspace (General tab) ────────────────────────────────────────

class WorkspaceSettings(BaseModel):
    organization_id: str
    name: str


class WorkspaceSettingsUpdate(BaseModel):
    name: str
    organization_id: str = DEFAULT_ORGANIZATION_ID


# ── Settings: Assistant config (Assistant tab) ───────────────────────────────

class AssistantConfig(BaseModel):
    organization_id: str
    name: str
    personality: str
    instructions: str


class AssistantConfigUpdate(BaseModel):
    name: str
    personality: str
    instructions: str
    organization_id: str = DEFAULT_ORGANIZATION_ID


# ── Settings: Usage ───────────────────────────────────────────────────────────

class UsageResponse(BaseModel):
    documents_used: int
    documents_limit: int
    questions_used: int
    questions_limit: int
    organization_id: str
    

class QuestionsOverTimePoint(BaseModel):
    date: str
    count: int
 
 
class PopularCollection(BaseModel):
    name: str
    percent: float
 
 
class AnalyticsSummary(BaseModel):
    questions_asked: int
    questions_asked_change_pct: float | None
    successful_answers: int
    success_rate_pct: float | None
    avg_response_time_ms: float | None
    avg_response_time_change_ms: float | None
    questions_over_time: list[QuestionsOverTimePoint]
    popular_collections: list[PopularCollection]
    organization_id: str