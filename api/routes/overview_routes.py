"""api/routes/overview_routes.py — Overview (dashboard) page."""

from fastapi import APIRouter

from api.controllers.overview_controller import get_stats_ctrl
from api.schemas.schemas import StatsResponse
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["overview"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(organization_id: str = DEFAULT_ORGANIZATION_ID):
    return await get_stats_ctrl(organization_id)
