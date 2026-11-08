"""api/routes/overview_routes.py — Overview (dashboard) page."""

from fastapi import APIRouter, Depends

from api.controllers.overview_controller import get_stats_ctrl
from api.schemas.schemas import StatsResponse
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["overview"])


@router.get("/stats", response_model=StatsResponse)
def get_stats(user: CurrentUser = Depends(get_current_user)):
    return get_stats_ctrl(user.organization_id)