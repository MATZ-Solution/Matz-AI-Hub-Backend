"""
api/helpers/activity.py
------------------------
Daily activity counts in the viewer's time zone (task B1), shared by the
Overview chart and the Analytics tab so both show the same numbers.

  questions = rows in chat_events (every attempt, failures included — B2)
  documents = rows in documents by created_at (deleted ones still count:
              they were added that day)

The window always ends today (local) and includes it.
"""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agents.langgraph_agent.utils import utils as db
from agents.langgraph_agent.utils.utils import _exec, _parse_ts

PAGE_SIZE = 1000  # Supabase returns at most 1,000 rows per request


def resolve_tz(name: str | None) -> ZoneInfo | timezone:
    """The IANA zone the browser reported, or UTC if it is missing or unknown."""
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return timezone.utc


def local_days(days: int, tz) -> list[date]:
    today = datetime.now(tz).date()
    return [today - timedelta(days=days - 1 - i) for i in range(days)]


def _created_at_since(table: str, organization_id: str, since: datetime) -> list[datetime]:
    client = db.get_supabase()
    out: list[datetime] = []
    start = 0
    while True:
        rows = _exec(
            client.table(table).select("created_at")
            .eq("organization_id", organization_id)
            .gte("created_at", since.isoformat())
            .order("created_at")
            .range(start, start + PAGE_SIZE - 1)
        ).data or []
        out.extend(_parse_ts(r["created_at"]) for r in rows)
        if len(rows) < PAGE_SIZE:
            return out
        start += PAGE_SIZE


def daily_activity(organization_id: str, days: int, tz_name: str | None) -> list[dict]:
    """[{date: 'YYYY-MM-DD', questions: n, documents: n}] for the last `days`
    local days, oldest first, today last. Raises if the database is unreachable."""
    tz = resolve_tz(tz_name)
    day_list = local_days(days, tz)
    since = datetime.combine(day_list[0], time.min, tzinfo=tz).astimezone(timezone.utc)
    buckets = {d: {"date": d.isoformat(), "questions": 0, "documents": 0} for d in day_list}
    for table, key in (("chat_events", "questions"), ("documents", "documents")):
        for ts in _created_at_since(table, organization_id, since):
            day = ts.astimezone(tz).date()
            if day in buckets:
                buckets[day][key] += 1
    return [buckets[d] for d in day_list]
