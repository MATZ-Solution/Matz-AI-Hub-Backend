"""api/routes/assistant_routes.py — Assistant page (chat + sessions)."""

from fastapi import APIRouter, Depends

from api.controllers.assistant_controller import (
    chat_ctrl, create_session_ctrl, list_sessions_ctrl,
    delete_session_ctrl, get_session_messages_ctrl,
)
from api.schemas.schemas import ChatRequest, ChatResponse
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["assistant"])


# Handlers are plain `def`, not `async def`: they call supabase-py and
# qdrant-client, which are synchronous. Inside `async def` those block the
# event loop and stall every other request; as `def`, FastAPI runs them in a
# threadpool where blocking is safe.

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, user: CurrentUser = Depends(get_current_user)):
    return chat_ctrl(request, user.organization_id, user_id=user.id)


@router.post("/sessions")
def create_chat_session(user: CurrentUser = Depends(get_current_user)):
    return create_session_ctrl(user.organization_id)


@router.get("/sessions")
def list_sessions(user: CurrentUser = Depends(get_current_user)):
    return list_sessions_ctrl(user.organization_id)


@router.delete("/sessions/{session_id}")
def delete_chat_session(session_id: str, user: CurrentUser = Depends(get_current_user)):
    return delete_session_ctrl(session_id, user.organization_id)


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str, user: CurrentUser = Depends(get_current_user)):
    return get_session_messages_ctrl(session_id, user.organization_id)