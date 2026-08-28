"""api/routes/assistant_routes.py — Assistant page (chat + sessions)."""

from fastapi import APIRouter

from api.controllers.assistant_controller import (
    chat_ctrl, create_session_ctrl, list_sessions_ctrl,
    delete_session_ctrl, get_session_messages_ctrl,
)
from api.schemas.schemas import ChatRequest, ChatResponse
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["assistant"])


# Handlers are plain `def`, not `async def`: they call supabase-py and
# qdrant-client, which are synchronous. Inside `async def` those block the
# event loop and stall every other request; as `def`, FastAPI runs them in a
# threadpool where blocking is safe.

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, organization_id: str = DEFAULT_ORGANIZATION_ID):
    return chat_ctrl(request, organization_id)


@router.post("/sessions")
def create_chat_session(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return create_session_ctrl(organization_id)


@router.get("/sessions")
def list_sessions(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return list_sessions_ctrl(organization_id)


@router.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str, organization_id: str = DEFAULT_ORGANIZATION_ID):
    return delete_session_ctrl(session_id, organization_id)


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str, organization_id: str = DEFAULT_ORGANIZATION_ID):
    return get_session_messages_ctrl(session_id, organization_id)