"""api/routes/settings_routes.py — Settings page (workspace, assistant config, members, usage)."""

from fastapi import APIRouter

from api.controllers.settings_controller import (
    get_workspace_settings_ctrl, update_workspace_settings_ctrl,
    get_assistant_config_ctrl, update_assistant_config_ctrl,
    get_members_ctrl, create_member_ctrl, delete_member_ctrl,
    get_usage_ctrl,
)
from api.schemas.schemas import (
    WorkspaceSettings, WorkspaceSettingsUpdate,
    AssistantConfig, AssistantConfigUpdate,
    MemberCreate, UsageResponse,
)
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["settings"])


@router.get("/settings/workspace", response_model=WorkspaceSettings)
async def get_workspace_settings_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await get_workspace_settings_ctrl(organization_id)


@router.put("/settings/workspace", response_model=WorkspaceSettings)
async def update_workspace_settings_route(request: WorkspaceSettingsUpdate):
    return await update_workspace_settings_ctrl(request)


@router.get("/settings/assistant", response_model=AssistantConfig)
async def get_assistant_config_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await get_assistant_config_ctrl(organization_id)


@router.put("/settings/assistant", response_model=AssistantConfig)
async def update_assistant_config_route(request: AssistantConfigUpdate):
    return await update_assistant_config_ctrl(request)


@router.get("/settings/members")
async def get_members_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await get_members_ctrl(organization_id)


@router.post("/settings/members")
async def create_member_route(request: MemberCreate):
    return await create_member_ctrl(request)


@router.delete("/settings/members/{member_id}")
async def delete_member_route(member_id: str):
    return await delete_member_ctrl(member_id)


@router.get("/settings/usage", response_model=UsageResponse)
async def get_usage_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await get_usage_ctrl(organization_id)