"""
api/controllers/settings_controller.py
------------------------------------------
Backs the Settings page: General (workspace identity), Assistant config,
Members, and Usage — all previously hardcoded on the frontend.
"""

from fastapi import HTTPException

from agent.src.utils.logger import logger
from agent.src.utils.supabase_client import (
    get_workspace_settings, upsert_workspace_settings,
    get_assistant_config, upsert_assistant_config,
    get_members, create_member, delete_member,
    count_user_questions,
)
from api.helpers.qdrant_helper import scroll_all_points
from api.config.settings import DEFAULT_ORGANIZATION_ID
from api.schemas.schemas import (
    WorkspaceSettings, WorkspaceSettingsUpdate,
    AssistantConfig, AssistantConfigUpdate,
    MemberCreate, UsageResponse,
)

# Fixed plan limits — there's no billing/plans system yet, so these stay
# constants until a real plans table is introduced.
DOCUMENTS_LIMIT = 500
QUESTIONS_LIMIT = 10000


# ── General (workspace) ──────────────────────────────────────────────────────

async def get_workspace_settings_ctrl(organization_id: str = DEFAULT_ORGANIZATION_ID) -> WorkspaceSettings:
    data = get_workspace_settings(organization_id)
    return WorkspaceSettings(**data)


async def update_workspace_settings_ctrl(request: WorkspaceSettingsUpdate) -> WorkspaceSettings:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Workspace name cannot be empty")
    result = upsert_workspace_settings(request.organization_id, request.name, request.url)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save workspace settings")
    logger.info("API → workspace settings updated: %s", request.organization_id)
    return WorkspaceSettings(**result)


# ── Assistant config ──────────────────────────────────────────────────────────

async def get_assistant_config_ctrl(organization_id: str = DEFAULT_ORGANIZATION_ID) -> AssistantConfig:
    data = get_assistant_config(organization_id)
    return AssistantConfig(**data)


async def update_assistant_config_ctrl(request: AssistantConfigUpdate) -> AssistantConfig:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Assistant name cannot be empty")
    result = upsert_assistant_config(request.organization_id, request.name, request.personality, request.instructions)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save assistant config")
    logger.info("API → assistant config updated: %s", request.organization_id)
    return AssistantConfig(**result)


# ── Members ───────────────────────────────────────────────────────────────────

async def get_members_ctrl(organization_id: str = DEFAULT_ORGANIZATION_ID) -> dict:
    members = get_members(organization_id)
    return {"members": members, "total": len(members)}


async def create_member_ctrl(request: MemberCreate) -> dict:
    if not request.name.strip() or not request.email.strip():
        raise HTTPException(status_code=400, detail="Name and email are required")
    result = create_member(request.organization_id, request.name, request.email, request.role)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to add member")
    logger.info("API → member added: %s", request.email)
    return {"success": True, "member": result}


async def delete_member_ctrl(member_id: str) -> dict:
    success = delete_member(member_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete member")
    return {"success": True, "member_id": member_id}


# ── Usage ─────────────────────────────────────────────────────────────────────

async def get_usage_ctrl(organization_id: str = DEFAULT_ORGANIZATION_ID) -> UsageResponse:
    try:
        all_points = scroll_all_points(organization_id)
        doc_ids = {p.payload.get("document_id", "unknown") for p in all_points}
        questions_used = count_user_questions(organization_id)
        return UsageResponse(
            documents_used=len(doc_ids),
            documents_limit=DOCUMENTS_LIMIT,
            questions_used=questions_used,
            questions_limit=QUESTIONS_LIMIT,
            organization_id=organization_id,
        )
    except Exception as e:
        logger.error("Usage error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))