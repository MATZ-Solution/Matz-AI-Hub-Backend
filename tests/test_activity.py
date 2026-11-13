"""
tests/test_activity.py — B1: real activity chart.

Done when: the chart's numbers match the Analytics tab for the same day,
and a new org shows the empty state (all zeros).

Run from Backend/:  .venv/bin/pytest tests/test_activity.py -q
"""

import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.helpers import activity
from tests.conftest import session_headers
from tests.fakes import FakeSupabase

ORG = "org-activity"
TZ = "Asia/Karachi"  # UTC+5, no DST


@pytest.fixture
def db(monkeypatch):
    fake = FakeSupabase()
    monkeypatch.setattr("agents.langgraph_agent.utils.utils.get_supabase", lambda: fake)
    return fake


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _client(org=ORG):
    return TestClient(app, raise_server_exceptions=False, headers=session_headers(org=org))


def test_new_org_gets_30_zero_days_ending_today(db):
    res = _client("org-brand-new").get("/activity", params={"tz": TZ})
    assert res.status_code == 200
    days = res.json()["days"]
    assert len(days) == 30
    assert all(d["questions"] == 0 and d["documents"] == 0 for d in days)
    assert days[-1]["date"] == datetime.now(activity.resolve_tz(TZ)).date().isoformat()


def test_counts_are_bucketed_in_the_viewers_time_zone(db):
    tz = activity.resolve_tz(TZ)
    today = datetime.now(tz).replace(hour=0, minute=30, second=0, microsecond=0)
    # 00:30 local today is still "yesterday" in UTC — it must count as today.
    db.tables["chat_events"] = [
        {"organization_id": ORG, "created_at": _ts(today)},
        {"organization_id": ORG, "created_at": _ts(today)},
        {"organization_id": ORG, "created_at": _ts(today - timedelta(days=3))},
        {"organization_id": "other-org", "created_at": _ts(today)},
        {"organization_id": ORG, "created_at": _ts(today - timedelta(days=45))},  # outside window
    ]
    db.tables["documents"] = [{"organization_id": ORG, "id": "d1", "created_at": _ts(today)}]
    days = _client().get("/activity", params={"tz": TZ}).json()["days"]
    assert days[-1] == {"date": today.date().isoformat(), "questions": 2, "documents": 1}
    assert days[-4]["questions"] == 1
    assert sum(d["questions"] for d in days) == 3


def test_chart_matches_analytics_for_the_same_day(db, monkeypatch):
    monkeypatch.setattr("api.controllers.analytics_controller.get_analytics_summary",
                        lambda org, days: _empty_summary(org, days))
    now = datetime.now(timezone.utc)
    db.tables["chat_events"] = [
        {"organization_id": ORG, "created_at": _ts(now - timedelta(days=d, minutes=1))}
        for d in (0, 0, 0, 1, 2, 2, 9, 20)
    ]
    chart = {d["date"]: d["questions"] for d in _client().get("/activity", params={"tz": TZ}).json()["days"]}
    summary = _client().get("/analytics/summary", params={"days": 14, "tz": TZ}).json()
    analytics = {p["date"]: p["count"] for p in summary["questions_over_time"]}
    assert len(analytics) == 14
    assert list(analytics)[-1] == list(chart)[-1], "analytics window must end today"
    for day, count in analytics.items():
        assert chart[day] == count
    assert summary["questions_asked"] == sum(analytics.values()) == 7
    # 1 question in the previous 14 days (day 20) → real change, not a dash.
    assert summary["questions_asked_change_pct"] == 600.0


def test_unknown_time_zone_falls_back_to_utc(db):
    res = _client().get("/activity", params={"tz": "Not/AZone"})
    assert res.status_code == 200
    assert res.json()["time_zone"] == "UTC"


def test_upload_and_delete_record_document_rows(db, monkeypatch):
    from api.controllers import knowledge_controller
    monkeypatch.setattr(knowledge_controller, "extract_text",
                        lambda path: {"success": True, "text": "text " * 50, "page_count": 1, "method": "txt"})
    monkeypatch.setattr(knowledge_controller, "ingest_chunks", lambda **k: ["v1"])
    res = _client().post("/ingest", files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")}, data={
        "document_title": "Leave", "collection_name": "HR", "collection_id": "c1", "document_id": "doc-9"})
    assert res.status_code == 200
    row = db.rows("documents", organization_id=ORG, id="doc-9")[0]
    assert row["created_by"] == "user-test-1" and row["version"] == 1 and row["deleted_at"] is None

    monkeypatch.setattr(knowledge_controller, "_require_document", lambda *a: None)
    monkeypatch.setattr(knowledge_controller, "delete_document_chunks", lambda *a: None)
    assert _client().delete("/documents/doc-9").status_code == 200
    assert db.rows("documents", organization_id=ORG, id="doc-9")[0]["deleted_at"] is not None
    # A deleted document still counts as added on its day.
    assert _client().get("/activity", params={"tz": TZ}).json()["days"][-1]["documents"] == 1


def test_paging_reads_past_1000_rows(db, monkeypatch):
    monkeypatch.setattr(activity, "PAGE_SIZE", 10)
    now = _ts(datetime.now(timezone.utc))
    db.tables["chat_events"] = [{"organization_id": ORG, "created_at": now} for _ in range(25)]
    assert activity.daily_activity(ORG, 30, "UTC")[-1]["questions"] == 25


def _empty_summary(org, days):
    return {
        "questions_asked": 0, "questions_asked_change_pct": None, "successful_answers": 0,
        "success_rate_pct": None, "success_rate_change_pct": None, "avg_response_time_ms": None,
        "avg_response_time_change_ms": None, "questions_over_time": [], "popular_collections": [],
        "organization_id": org,
    }
