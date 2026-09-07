"""
api/controllers/settings_controller.py
------------------------------------------
Backs the Settings page: General (workspace identity), Assistant config,
and Usage — all previously hardcoded on the frontend.
"""

from fastapi import HTTPException

from agents.langgraph_agent.utils.utils import logger
from agents.langgraph_agent.utils.utils import (
    get_workspace_settings, upsert_workspace_settings,
    get_assistant_config, upsert_assistant_config,
    count_user_questions,
)
from agents.langgraph_agent.utils.utils import invalidate_assistant_config
from api.helpers.qdrant_helper import scroll_all_points
from api.schemas.schemas import (
    WorkspaceSettings, WorkspaceSettingsUpdate,
    AssistantConfig, AssistantConfigUpdate,
    UsageResponse,
)

# Fixed plan limits — there's no billing/plans system yet, so these stay
# constants until a real plans table is introduced.
DOCUMENTS_LIMIT = 500
QUESTIONS_LIMIT = 10000


# ── General (workspace) ──────────────────────────────────────────────────────

def get_workspace_settings_ctrl(organization_id: str) -> WorkspaceSettings:
    data = get_workspace_settings(organization_id)
    return WorkspaceSettings(**data)


def update_workspace_settings_ctrl(request: WorkspaceSettingsUpdate, organization_id: str) -> WorkspaceSettings:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Workspace name cannot be empty")
    result = upsert_workspace_settings(organization_id, request.name)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save workspace settings")
    logger.info("API → workspace settings updated: %s", organization_id)
    return WorkspaceSettings(**result)


# ── Assistant config ──────────────────────────────────────────────────────────

def get_assistant_config_ctrl(organization_id: str) -> AssistantConfig:
    data = get_assistant_config(organization_id)
    return AssistantConfig(**data)


def update_assistant_config_ctrl(request: AssistantConfigUpdate, organization_id: str) -> AssistantConfig:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Assistant name cannot be empty")
    result = upsert_assistant_config(organization_id, request.name, request.personality, request.instructions)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save assistant config")

    # Drop the cached copy so the very next chat message builds its system
    # prompt from these new values instead of the stale ones.
    invalidate_assistant_config(organization_id)

    logger.info("API → assistant config updated: %s", organization_id)
    return AssistantConfig(**result)



# ── Usage ─────────────────────────────────────────────────────────────────────

def get_usage_ctrl(organization_id: str) -> UsageResponse:
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