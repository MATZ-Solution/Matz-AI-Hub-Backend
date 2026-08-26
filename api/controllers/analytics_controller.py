"""
api/controllers/analytics_controller.py
-------------------------------------------
Backs the Analytics page: real usage/performance metrics computed from
Supabase message and session history (see get_analytics_summary in
agent/src/utils/supabase_client.py for the actual query logic).
"""

from fastapi import HTTPException

from agent.src.utils.logger import logger
from agent.src.utils.supabase_client import get_analytics_summary
from api.config.settings import DEFAULT_ORGANIZATION_ID
from api.schemas.schemas import AnalyticsSummary

DEFAULT_LOOKBACK_DAYS = 14


async def get_analytics_ctrl(
    organization_id: str = DEFAULT_ORGANIZATION_ID,
    days: int = DEFAULT_LOOKBACK_DAYS,
) -> AnalyticsSummary:
    try:
        data = get_analytics_summary(organization_id, days)
        return AnalyticsSummary(**data)
    except Exception as e:
        logger.error("Analytics error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))