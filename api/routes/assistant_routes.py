"""api/routes/assistant_routes.py — Assistant page (chat + sessions)."""

from fastapi import APIRouter

from api.controllers.assistant_controller import (
    chat_ctrl, create_session_ctrl, list_sessions_ctrl,
    delete_session_ctrl, get_session_messages_ctrl,
)
from api.schemas.schemas import ChatRequest, ChatResponse
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["assistant"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return await chat_ctrl(request)


@router.post("/sessions")
async def create_chat_session(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await create_session_ctrl(organization_id)


@router.get("/sessions")
async def list_sessions(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await list_sessions_ctrl(organization_id)


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    return await delete_session_ctrl(session_id)


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    return await get_session_messages_ctrl(session_id)
