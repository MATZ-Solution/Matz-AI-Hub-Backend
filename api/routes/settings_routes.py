"""api/routes/settings_routes.py — Settings page (workspace, assistant config, usage)."""

from fastapi import APIRouter, Depends

from api.controllers.settings_controller import (
    get_workspace_settings_ctrl, update_workspace_settings_ctrl,
    get_assistant_config_ctrl, update_assistant_config_ctrl,
    get_usage_ctrl,
)
from api.schemas.schemas import (
    WorkspaceSettings, WorkspaceSettingsUpdate,
    AssistantConfig, AssistantConfigUpdate,
    UsageResponse,
)
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["settings"])


@router.get("/settings/workspace", response_model=WorkspaceSettings)
def get_workspace_settings_route(user: CurrentUser = Depends(get_current_user)):
    return get_workspace_settings_ctrl(user.organization_id)


@router.put("/settings/workspace", response_model=WorkspaceSettings)
def update_workspace_settings_route(
    request: WorkspaceSettingsUpdate,
    user: CurrentUser = Depends(get_current_user),
):
    return update_workspace_settings_ctrl(request, user.organization_id)


@router.get("/settings/assistant", response_model=AssistantConfig)
def get_assistant_config_route(user: CurrentUser = Depends(get_current_user)):
    return get_assistant_config_ctrl(user.organization_id)


@router.put("/settings/assistant", response_model=AssistantConfig)
def update_assistant_config_route(
    request: AssistantConfigUpdate,
    user: CurrentUser = Depends(get_current_user),
):
    return update_assistant_config_ctrl(request, user.organization_id)


@router.get("/settings/usage", response_model=UsageResponse)
def get_usage_route(user: CurrentUser = Depends(get_current_user)):
    return get_usage_ctrl(user.organization_id)