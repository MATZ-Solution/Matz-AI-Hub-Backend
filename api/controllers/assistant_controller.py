"""
api/controllers/assistant_controller.py
------------------------------------------
Backs the Assistant page: chat + chat sessions.

*** This is where the API integrates with the LangGraph agent (Backend/agent). ***
`chat()` imported below from agents.langgraph_agent.main_langgraph_agent is the agent's entrypoint —
everything upstream (routing, retrieval, grading, generation) happens inside
Backend/agent and is opaque to the API layer. This controller's job is just to
call it, time it, and persist the turn to Supabase.
"""

import time

from fastapi import HTTPException

from agents.langgraph_agent.main_langgraph_agent import chat as run_agent_chat   # ← agent <-> backend integration point
from agents.langgraph_agent.utils.utils import logger
from agents.langgraph_agent.utils.utils import (
    create_session, get_sessions, delete_session,
    save_message, get_messages,
)
from api.schemas.schemas import ChatRequest, ChatResponse, CitationResponse


def chat_ctrl(request: ChatRequest, organization_id: str) -> ChatResponse:
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="user_query cannot be empty")

    logger.info("API → /chat | org: %s | query: %s", organization_id, request.user_query[:50])
    start_time = time.time()

    try:
        answer, citations, _, is_grounded = run_agent_chat(
            user_query=request.user_query,
            chat_history=request.chat_history,
            organization_id=organization_id,
        )
    except Exception as e:
        logger.error("Agent error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    response_time_ms = int((time.time() - start_time) * 1000)

    if request.session_id:
        try:
            save_message(request.session_id, "user", request.user_query, [])
            # is_grounded + response_time_ms feed the Analytics page's
            # "Successful answers" and "Avg. response time" metrics.
            save_message(
                request.session_id, "assistant", answer, citations,
                is_grounded=is_grounded, response_time_ms=response_time_ms,
            )
        except Exception as e:
            logger.warning("Supabase → failed to save messages: %s", e)

    logger.info("API → /chat done | %dms | %d citations", response_time_ms, len(citations))
    return ChatResponse(
        answer=answer,
        citations=[CitationResponse(**c) for c in citations],
        conversation_id=request.conversation_id,
        session_id=request.session_id,
        response_time_ms=response_time_ms,
        organization_id=organization_id,
    )


def create_session_ctrl(organization_id: str) -> dict:
    session_id = create_session(organization_id)
    if not session_id:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return {"session_id": session_id}


def list_sessions_ctrl(organization_id: str) -> dict:
    all_sessions = get_sessions(organization_id)
    return {"sessions": all_sessions, "total": len(all_sessions)}


def delete_session_ctrl(session_id: str, organization_id: str) -> dict:
    success = delete_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session")
    return {"success": True, "session_id": session_id}


def get_session_messages_ctrl(session_id: str, organization_id: str) -> dict:
    messages = get_messages(session_id)
    return {"messages": messages, "session_id": session_id}