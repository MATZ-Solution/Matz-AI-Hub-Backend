"""api/routes/settings_routes.py — Settings page (workspace, assistant config, usage)."""

from fastapi import APIRouter

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
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["settings"])


@router.get("/settings/workspace", response_model=WorkspaceSettings)
def get_workspace_settings_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return get_workspace_settings_ctrl(organization_id)


@router.put("/settings/workspace", response_model=WorkspaceSettings)
def update_workspace_settings_route(
    request: WorkspaceSettingsUpdate,
    organization_id: str = DEFAULT_ORGANIZATION_ID,
):
    return update_workspace_settings_ctrl(request, organization_id)


@router.get("/settings/assistant", response_model=AssistantConfig)
def get_assistant_config_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return get_assistant_config_ctrl(organization_id)


@router.put("/settings/assistant", response_model=AssistantConfig)
def update_assistant_config_route(
    request: AssistantConfigUpdate,
    organization_id: str = DEFAULT_ORGANIZATION_ID,
):
    return update_assistant_config_ctrl(request, organization_id)


@router.get("/settings/usage", response_model=UsageResponse)
def get_usage_route(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return get_usage_ctrl(organization_id)