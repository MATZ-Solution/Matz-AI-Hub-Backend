"""api/routes/analytics_routes.py — Analytics page (usage & performance)."""

from fastapi import APIRouter, Depends

from api.controllers.analytics_controller import get_analytics_ctrl, DEFAULT_LOOKBACK_DAYS
from api.schemas.schemas import AnalyticsSummary
from api.auth import CurrentUser, get_current_user

router = APIRouter(tags=["analytics"])


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary_route(
    user: CurrentUser = Depends(get_current_user),
    days: int = DEFAULT_LOOKBACK_DAYS,
):
    return get_analytics_ctrl(user.organization_id, days)