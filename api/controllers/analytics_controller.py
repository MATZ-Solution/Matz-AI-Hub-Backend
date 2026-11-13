"""
api/controllers/analytics_controller.py
-------------------------------------------
Backs the Analytics page: real usage/performance metrics computed from
Supabase message and session history (see get_analytics_summary in
agent/src/utils/supabase_client.py for the actual query logic).
"""

from fastapi import HTTPException

from agents.langgraph_agent.utils.utils import logger, _pct_change
from agents.langgraph_agent.utils.utils import get_analytics_summary
from api.helpers.activity import daily_activity
from api.schemas.schemas import AnalyticsSummary

DEFAULT_LOOKBACK_DAYS = 14


def get_analytics_ctrl(
    organization_id: str,
    days: int = DEFAULT_LOOKBACK_DAYS,
    tz: str | None = None,
) -> AnalyticsSummary:
    try:
        data = get_analytics_summary(organization_id, days)
        # Questions come from chat_events (every attempt, failures included)
        # in the viewer's time zone, ending today — the same counts as the
        # Overview activity chart (B1). Two windows: previous, then current.
        points = daily_activity(organization_id, days * 2, tz)
        previous, current = points[:days], points[days:]
        asked = sum(p["questions"] for p in current)
        data["questions_asked"] = asked
        data["questions_asked_change_pct"] = _pct_change(sum(p["questions"] for p in previous), asked)
        data["questions_over_time"] = [{"date": p["date"], "count": p["questions"]} for p in current]
        return AnalyticsSummary(**data)
    except Exception as e:
        logger.error("Analytics error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
