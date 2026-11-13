"""api/routes/overview_routes.py — Overview (dashboard) page."""

from fastapi import APIRouter, Depends, Query

from api.controllers.overview_controller import get_stats_ctrl
from api.schemas.schemas import StatsResponse, ActivityResponse
from api.controllers.overview_controller import get_activity_ctrl
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["overview"])


@router.get("/stats", response_model=StatsResponse)
def get_stats(user: CurrentUser = Depends(get_current_user)):
    return get_stats_ctrl(user.organization_id)


@router.get("/activity", response_model=ActivityResponse)
def get_activity(
    user: CurrentUser = Depends(get_current_user),
    days: int = Query(30, ge=1, le=90),
    tz: str | None = Query(None, max_length=64, description="IANA time zone of the viewer, e.g. Asia/Karachi"),
):
    """Daily questions and documents added (B1), in the viewer's time zone."""
    return get_activity_ctrl(user.organization_id, days, tz)
