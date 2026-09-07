"""
api/controllers/analytics_controller.py
-------------------------------------------
Backs the Analytics page: real usage/performance metrics computed from
Supabase message and session history (see get_analytics_summary in
agent/src/utils/supabase_client.py for the actual query logic).
"""

from fastapi import HTTPException

from agents.langgraph_agent.utils.utils import logger
from agents.langgraph_agent.utils.utils import get_analytics_summary
from api.schemas.schemas import AnalyticsSummary

DEFAULT_LOOKBACK_DAYS = 14


def get_analytics_ctrl(
    organization_id: str,
    days: int = DEFAULT_LOOKBACK_DAYS,
) -> AnalyticsSummary:
    try:
        data = get_analytics_summary(organization_id, days)
        return AnalyticsSummary(**data)
    except Exception as e:
        logger.error("Analytics error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))