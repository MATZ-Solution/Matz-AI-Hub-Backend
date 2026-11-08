"""
tests/fakes.py — in-memory stand-ins for Qdrant and Supabase.

They implement only what the code uses, with the same filtering semantics,
so the real controllers, store and agent nodes run unchanged against them.
Used by the tenant-isolation suite (F1), which must run in CI without any
real service or secret.
"""

import copy
import uuid
from datetime import datetime, timezone

from qdrant_client.models import Filter


# ── Qdrant ───────────────────────────────────────────────────────────────────

def _matches(payload: dict, flt: Filter | None) -> bool:
    if flt is None:
        return True
    return all(payload.get(c.key) == c.match.value for c in (flt.must or []))


class _Hit:
    def __init__(self, point, score):
        self.id, self.payload, self.score = point["id"], copy.deepcopy(point["payload"]), score


class FakeQdrant:
    """Points are dicts {id, vector, payload}. Every point scores the same, so
    a query returns whatever the filter lets through — exactly what a
    cross-tenant leak would look like."""

    def __init__(self):
        self.points: dict = {}

    # writes
    def upsert(self, collection_name, points):
        for p in points:
            self.points[p.id] = {"id": p.id, "vector": p.vector, "payload": dict(p.payload)}

    def delete(self, collection_name, points_selector):
        for pid in [pid for pid, p in self.points.items() if _matches(p["payload"], points_selector)]:
            del self.points[pid]

    # reads
    def query_points(self, collection_name, query, query_filter=None, limit=10, with_payload=True):
        hits = [_Hit(p, 0.9) for p in self.points.values() if _matches(p["payload"], query_filter)]
        return type("QueryResponse", (), {"points": hits[:limit]})()

    def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None,
               with_payload=True, with_vectors=False):
        matching = [_Hit(p, 1.0) for p in self.points.values() if _matches(p["payload"], scroll_filter)]
        start = offset or 0
        page = matching[start:start + limit]
        next_offset = start + limit if start + limit < len(matching) else None
        return page, next_offset

    def get_collection(self, collection_name):
        return {}

    def create_payload_index(self, **_):
        return None

    # test helpers
    def payloads(self, **where):
        return [p["payload"] for p in self.points.values()
                if all(p["payload"].get(k) == v for k, v in where.items())]


# ── Supabase ─────────────────────────────────────────────────────────────────

_UNIQUE_KEYS = {"workspace_settings": "organization_id", "assistant_config": "organization_id"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Query:
    def __init__(self, db, table):
        self.db, self.table = db, table
        self.op, self.payload = "select", None
        self.filters, self.order_by, self.limit_n = [], None, None

    # verbs
    def select(self, *_cols, **_kw):
        self.op = "select"; return self

    def insert(self, rows):
        self.op, self.payload = "insert", rows; return self

    def upsert(self, row, **_kw):
        self.op, self.payload = "upsert", row; return self

    def update(self, values):
        self.op, self.payload = "update", values; return self

    def delete(self):
        self.op = "delete"; return self

    # modifiers
    def eq(self, key, value):
        self.filters.append(lambda r: str(r.get(key)) == str(value)); return self

    def in_(self, key, values):
        allowed = {str(v) for v in values}
        self.filters.append(lambda r: str(r.get(key)) in allowed); return self

    def order(self, key, desc=False):
        self.order_by = (key, desc); return self

    def limit(self, n):
        self.limit_n = n; return self

    # run
    def _rows(self):
        rows = [r for r in self.db.tables.setdefault(self.table, []) if all(f(r) for f in self.filters)]
        if self.order_by:
            key, desc = self.order_by
            rows.sort(key=lambda r: r.get(key) or "", reverse=desc)
        return rows[: self.limit_n] if self.limit_n else rows

    def execute(self):
        table = self.db.tables.setdefault(self.table, [])
        if self.op == "select":
            data = copy.deepcopy(self._rows())
        elif self.op == "insert":
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            data = []
            for row in rows:
                full = {"id": str(uuid.uuid4()), "created_at": _now(), **row}
                table.append(full); data.append(copy.deepcopy(full))
        elif self.op == "upsert":
            key = _UNIQUE_KEYS.get(self.table, "id")
            existing = next((r for r in table if r.get(key) == self.payload.get(key)), None)
            if existing:
                existing.update(self.payload); data = [copy.deepcopy(existing)]
            else:
                full = {"created_at": _now(), **self.payload}
                table.append(full); data = [copy.deepcopy(full)]
        elif self.op == "update":
            rows = self._rows()
            for r in rows:
                r.update(self.payload)
            data = copy.deepcopy(rows)
        elif self.op == "delete":
            doomed = self._rows()
            self.db.tables[self.table] = [r for r in table if r not in doomed]
            if self.table == "sessions":  # ON DELETE CASCADE to messages
                ids = {r["id"] for r in doomed}
                self.db.tables["messages"] = [m for m in self.db.tables.get("messages", []) if m["session_id"] not in ids]
            data = copy.deepcopy(doomed)
        return type("Result", (), {"data": data})()


class _Rpc:
    def __init__(self, db, name, params):
        self.db, self.name, self.params = db, name, params

    def execute(self):
        org = self.params["org_id"]
        sessions = {s["id"] for s in self.db.tables.get("sessions", []) if s["organization_id"] == org}
        msgs = [m for m in self.db.tables.get("messages", []) if m["session_id"] in sessions]
        if self.name == "count_org_user_questions":
            data = sum(1 for m in msgs if m["role"] == "user")
        elif self.name == "org_messages_since":
            data = sorted((m for m in msgs if m["created_at"] >= self.params["since"]),
                          key=lambda m: m["created_at"])
        else:
            raise NotImplementedError(self.name)
        return type("Result", (), {"data": copy.deepcopy(data)})()


class FakeSupabase:
    def __init__(self):
        self.tables: dict[str, list] = {}

    def table(self, name):
        return _Query(self, name)

    def rpc(self, name, params):
        return _Rpc(self, name, params)

    def rows(self, table, **where):
        return [r for r in self.tables.get(table, []) if all(r.get(k) == v for k, v in where.items())]




def fake_vector(_text: str) -> list[float]:
    """Deterministic 384-d vector; content does not matter to FakeQdrant."""
    return [0.01] * 384
