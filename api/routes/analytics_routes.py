"""api/routes/analytics_routes.py — Analytics page (usage & performance)."""

from fastapi import APIRouter

from api.controllers.analytics_controller import get_analytics_ctrl, DEFAULT_LOOKBACK_DAYS
from api.schemas.schemas import AnalyticsSummary
from api.config.settings import DEFAULT_ORGANIZATION_ID

router = APIRouter(tags=["analytics"])


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary_route(
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    days: int = DEFAULT_LOOKBACK_DAYS,
):
    return get_analytics_ctrl(organization_id, days)