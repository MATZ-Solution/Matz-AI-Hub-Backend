"""
tests/test_abuse_protection.py — A5: rate limits and server-side upload checks.

Done when: the 21st chat message within a minute returns rate_limited with a
friendly message, and a renamed .exe is rejected.

Run from Backend/:  .venv/bin/pytest tests/test_abuse_protection.py -q
"""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.controllers import assistant_controller, knowledge_controller
from api.helpers.file_helper import sanitize_filename
from api.rate_limit import IP_LIMIT
from tests.conftest import session_headers

client = TestClient(app, raise_server_exceptions=False, headers=session_headers())

ANSWER = {"answer": "20 days [1]", "citations": [], "chat_history": [], "is_grounded": True}


@pytest.fixture
def events(monkeypatch):
    rows = []
    monkeypatch.setattr(assistant_controller, "record_chat_event", rows.append)
    monkeypatch.setattr(assistant_controller, "run_agent_chat", lambda **_: ANSWER)
    return rows


# ── Rate limits ──────────────────────────────────────────────────────────────

def test_21st_chat_message_in_a_minute_is_rate_limited(events):
    for i in range(20):
        assert client.post("/chat", json={"user_query": f"q{i}"}).status_code == 200
    res = client.post("/chat", json={"user_query": "one too many"})
    assert res.status_code == 429
    body = res.json()
    assert body["code"] == "rate_limited"
    assert body["retryable"] is True
    assert body["message"] == "You're sending messages too quickly. Wait a minute and try again."
    # B2: the refused attempt is still logged.
    assert events[-1]["outcome"] == "rate_limited"
    assert len(events) == 21


def test_chat_limit_is_per_user(events):
    for i in range(20):
        client.post("/chat", json={"user_query": f"q{i}"})
    other = TestClient(app, raise_server_exceptions=False, headers=session_headers(user_id="user-test-2"))
    assert other.post("/chat", json={"user_query": "hi"}).status_code == 200


def test_search_limit(monkeypatch):
    from api.routes import knowledge_routes
    monkeypatch.setattr(knowledge_routes, "search_content_ctrl",
                        lambda *a, **k: {"results": [], "query": "x", "total": 0})
    for _ in range(60):
        assert client.post("/search", json={"query": "leave"}).status_code == 200
    assert client.post("/search", json={"query": "leave"}).json()["code"] == "rate_limited"


def test_upload_limit(monkeypatch):
    from api.routes import knowledge_routes
    monkeypatch.setattr(knowledge_routes, "ingest_document_ctrl", lambda *a, **k: {
        "success": True, "document_id": "d", "chunks_created": 1, "page_count": 1,
        "extraction_method": "txt", "vector_ids": []})
    form = {"document_title": "t", "collection_name": "HR", "collection_id": "c", "document_id": "d"}
    for _ in range(10):
        res = client.post("/ingest", files={"file": ("a.txt", b"hello", "text/plain")}, data=form)
        assert res.status_code == 200
    res = client.post("/ingest", files={"file": ("a.txt", b"hello", "text/plain")}, data=form)
    assert res.json()["code"] == "rate_limited"
    assert "10 uploads an hour" in res.json()["message"]


def test_ip_limit_applies_without_a_session():
    anon = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Forwarded-For": "203.0.113.7"}
    codes = [anon.get("/documents", headers=headers).status_code for _ in range(IP_LIMIT[0] + 1)]
    assert codes[:-1] == [401] * IP_LIMIT[0]
    assert codes[-1] == 429
    # Another IP is unaffected, and /health is never limited.
    assert anon.get("/documents", headers={"X-Forwarded-For": "203.0.113.8"}).status_code == 401
    assert anon.get("/health", headers=headers).status_code != 429


# ── Upload validation ────────────────────────────────────────────────────────

FORM = {"document_title": "Doc", "collection_name": "HR", "collection_id": "col-HR", "document_id": "doc-1"}


def _upload(name: str, content: bytes):
    return client.post("/ingest", files={"file": (name, io.BytesIO(content), "application/octet-stream")}, data=FORM)


def _ooxml(main_part: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr(main_part, "<doc/>")
    return buf.getvalue()


@pytest.fixture
def accept_pipeline(monkeypatch):
    """Lets a valid file through extraction and ingestion without real services."""
    monkeypatch.setattr(knowledge_controller, "extract_text",
                        lambda path: {"success": True, "text": "text " * 50, "page_count": 1, "method": "fake"})
    monkeypatch.setattr(knowledge_controller, "ingest_chunks", lambda **k: ["v1"])


def _assert_invalid(res, fragment: str):
    assert res.status_code == 400, res.text
    body = res.json()
    assert body["code"] == "invalid_file"
    assert fragment in body["message"]


def test_renamed_exe_is_rejected():
    exe = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 200
    for name in ("setup.pdf", "setup.docx", "setup.pptx", "setup.xlsx"):
        _assert_invalid(_upload(name, exe), "isn't a valid")
    _assert_invalid(_upload("setup.txt", exe), "isn't a plain text file")


def test_disallowed_extension_is_rejected():
    _assert_invalid(_upload("setup.exe", b"MZ"), "isn't supported")
    _assert_invalid(_upload("noextension", b"hello"), "isn't supported")


def test_empty_file_is_rejected():
    _assert_invalid(_upload("empty.txt", b""), "empty")


def test_oversized_file_is_rejected(monkeypatch):
    from api.helpers import file_helper
    monkeypatch.setattr(file_helper, "MAX_UPLOAD_BYTES", 1000)
    _assert_invalid(_upload("big.txt", b"a" * 1001), "larger than")


def test_password_protected_office_file_is_rejected():
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 500
    _assert_invalid(_upload("secret.docx", ole), "password-protected")


def test_password_protected_pdf_is_rejected():
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="hunter2", owner_password="owner")
    buf = io.BytesIO()
    writer.write(buf)
    _assert_invalid(_upload("secret.pdf", buf.getvalue()), "password-protected")


def test_zip_without_office_content_is_rejected():
    _assert_invalid(_upload("fake.docx", _ooxml("something/else.xml")), "isn't a valid DOCX")


def test_valid_files_are_accepted(accept_pipeline):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf = io.BytesIO()
    writer.write(pdf)
    cases = {
        "a.pdf": pdf.getvalue(),
        "a.docx": _ooxml("word/document.xml"),
        "a.pptx": _ooxml("ppt/presentation.xml"),
        "a.xlsx": _ooxml("xl/workbook.xml"),
        "a.txt": "Annual leave: 20 days — café".encode("utf-8"),
    }
    for name, content in cases.items():
        res = _upload(name, content)
        assert res.status_code == 200, (name, res.text)


def test_rejected_replacement_keeps_the_old_chunks(monkeypatch):
    deleted = []
    monkeypatch.setattr(knowledge_controller, "_require_document", lambda *a: None)
    monkeypatch.setattr(knowledge_controller, "delete_document_chunks", lambda *a: deleted.append(a))
    res = client.put("/documents/doc-1", files={"file": ("x.pdf", io.BytesIO(b"MZ junk"), "application/pdf")},
                     data={"document_title": "Doc", "collection_name": "HR", "collection_id": "col-HR"})
    assert res.status_code == 400
    assert deleted == []


@pytest.mark.parametrize("raw, expected", [
    ("../../etc/passwd.txt", "passwd.txt"),
    ("C:\\Users\\me\\Leave Policy.pdf", "Leave Policy.pdf"),
    ("report<script>.pdf", "report_script_.pdf"),
    ("bad\x00name\n.txt", "bad_name_.txt"),
    ("", "document"),
    ("..", "document"),
    ("a" * 300 + ".docx", "a" * 115 + ".docx"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected
