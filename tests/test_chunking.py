"""
tests/test_chunking.py — C1: structure-aware chunking.

Done when: the sample HR document yields more chunks than pages, and every
chunk has page and heading metadata.

The sample is built here as a 3-page PDF shaped like the live HR_Policy
document: a contents page, then numbered sections including "8. Annual Leave".

Run from Backend/:  .venv/bin/pytest tests/test_chunking.py -q
"""

import pytest

from data.pipeline.chunker import (
    MAX_TOKENS, OVERLAP_TOKENS, chunk_text, estimate_tokens, heading_level, split_blocks,
)
from data.pipeline.processor import extract_text

FILLER = ("Employees should read this section carefully and contact Human Resources "
          "with any questions about how it applies to their role. ")

HR_PAGES = [
    [
        "HR POLICY HANDBOOK",
        "Contents",
        "1. Introduction ........ 2",
        "2. Working Hours ........ 2",
        "8. Annual Leave ........ 3",
        "9. Sick Leave ........ 3",
    ],
    [
        "1. Introduction",
        "This handbook sets out the policies that apply to all permanent employees. " + FILLER,
        "2. Working Hours",
        "Standard working hours are 9:00 to 17:30, Monday to Friday. " + FILLER,
        "2.1 Overtime",
        "Overtime must be approved in advance by a line manager. " + FILLER * 2,
    ],
    [
        "8. Annual Leave",
        "Permanent employees receive 20 paid annual leave days per year. " + FILLER,
        "8.1 Carry Over",
        "Up to 5 unused days may be carried over to the next year. " + FILLER,
        "9. Sick Leave",
        "Employees receive 10 paid sick days per year with a medical certificate. " + FILLER,
    ],
]


@pytest.fixture
def hr_pdf(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    for lines in HR_PAGES:
        page = doc.new_page()
        rect = pymupdf.Rect(50, 50, 545, 800)
        page.insert_textbox(rect, "\n".join(lines), fontsize=10)
    path = tmp_path / "HR_Policy.pdf"
    doc.save(str(path))
    return str(path)


def _chunks_for(path, title="HR_Policy"):
    result = extract_text(path)
    assert result["success"], result
    return result, chunk_text(result["text"], "doc-hr", result["page_count"],
                              blocks=result.get("blocks"), title=title)


def test_sample_hr_document_yields_more_chunks_than_pages_with_metadata(hr_pdf):
    result, chunks = _chunks_for(hr_pdf)
    assert result["page_count"] == 3
    assert len(chunks) > result["page_count"]
    for c in chunks:
        assert 1 <= c["page_number"] <= c["page_end"] <= 3
        assert c["heading"]
        assert c["content"].startswith(c["heading"] + "\n\n")
        assert c["token_count"] <= MAX_TOKENS


def test_sections_get_their_heading_path_and_page(hr_pdf):
    _, chunks = _chunks_for(hr_pdf)
    by_heading = {c["heading"]: c for c in chunks}
    leave = by_heading["8. Annual Leave"]
    assert "20 paid annual leave days" in leave["content"]
    assert leave["page_number"] == 3
    assert "Up to 5 unused days" in by_heading["8. Annual Leave > 8.1 Carry Over"]["content"]
    assert "Overtime must be approved" in by_heading["2. Working Hours > 2.1 Overtime"]["content"]
    # The contents page is not indexed (C2): it would outrank the sections.
    assert not any("........" in c["content"] or "Contents" in c["content"] for c in chunks)
    # A chunk never spans two sections.
    assert "Sick Leave" not in leave["content"].split("\n\n", 1)[1]


def test_long_section_is_split_with_overlap_and_size_limits():
    sentences = [f"Rule {i} says employees must follow procedure number {i} at all times." for i in range(300)]
    text = "5. Procedures\n" + "\n\n".join(" ".join(sentences[i:i + 5]) for i in range(0, 300, 5))
    chunks = chunk_text(text, "doc-x", 1, title="Manual")
    assert len(chunks) >= 3
    sizes = [c["token_count"] for c in chunks]
    assert max(sizes) <= MAX_TOKENS
    assert min(sizes[:-1]) >= 400, sizes
    for a, b in zip(chunks, chunks[1:]):
        body_a = a["content"].split("\n\n", 1)[1]
        body_b = b["content"].split("\n\n", 1)[1]
        first_sentence_b = body_b.split(". ")[0]
        assert first_sentence_b in body_a, "next chunk must start with overlap from the previous one"
    # Overlap is about 80 tokens, never the whole chunk.
    overlap = chunks[1]["content"].split("\n\n", 1)[1]
    assert estimate_tokens(overlap) > OVERLAP_TOKENS


def test_docx_headings_come_from_styles(tmp_path):
    from docx import Document
    d = Document()
    d.add_heading("Leave Policy", level=1)
    d.add_paragraph("Staff get 20 days. " * 10)
    d.add_heading("Public Holidays", level=2)
    d.add_paragraph("There are 12 public holidays. " * 10)
    d.add_page_break()
    d.add_heading("Expenses", level=1)
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Item", "Limit"
    table.cell(1, 0).text, table.cell(1, 1).text = "Hotel", "150 per night"
    path = tmp_path / "policy.docx"
    d.save(str(path))
    _, chunks = _chunks_for(str(path), title="Policy")
    headings = [c["heading"] for c in chunks]
    assert headings == ["Leave Policy", "Leave Policy > Public Holidays", "Expenses"]
    assert chunks[2]["page_number"] == 2
    assert "Hotel | 150 per night" in chunks[2]["content"]


def test_pptx_slide_titles_are_headings(tmp_path):
    from pptx import Presentation
    prs = Presentation()
    for title, body in [("Onboarding", "Day one: collect your laptop from the IT desk."),
                        ("Security", "Lock your screen whenever you leave your desk.")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    path = tmp_path / "deck.pptx"
    prs.save(str(path))
    result = extract_text(str(path))
    chunks = chunk_text(result["text"], "d", result["page_count"], blocks=result["blocks"], title="Deck")
    assert [(c["heading"], c["page_number"]) for c in chunks] == [("Onboarding", 1), ("Security", 2)]


def test_text_before_any_heading_uses_the_title():
    chunks = chunk_text("Just a short note without headings.", "d", 1, title="Note")
    assert chunks[0]["heading"] == "Note"
    assert chunks[0]["content"] == "Note\n\nJust a short note without headings."


@pytest.mark.parametrize("line, toc", [
    ("8. Annual Leave ........ 3", True),
    ("Annual Leave …… 3", True),
    ("8. Annual Leave 3", True),
    ("Table of Contents", True),
    ("Contents", True),
    ("8. Annual Leave", False),
    ("Employees get 20 days per year 2024", False),
    ("Staff receive 5 days.", False),
])
def test_contents_lines(line, toc):
    from data.pipeline.chunker import is_contents_line
    assert is_contents_line(line) is toc


@pytest.mark.parametrize("line, level", [
    ("8. Annual Leave", 1),
    ("8.1 Carry Over", 2),
    ("2) Scope", 1),
    ("## Benefits", 2),
    ("ANNUAL LEAVE POLICY", 1),
    ("8. Annual Leave ........ 3", 0),     # contents entry
    ("8. Annual Leave 3", 0),              # contents entry without leaders
    ("1. Submit the form to HR.", 0),      # list item (sentence)
    ("Permanent employees receive 20 days", 0),
    ("NOTE", 0),                           # single word
    ("", 0),
])
def test_heading_level(line, level):
    assert heading_level(line) == level


def test_form_feed_starts_a_new_page():
    blocks = split_blocks("1. One\nfirst page text\f2. Two\nsecond page text")
    assert [(b["text"], b["page"]) for b in blocks] == [
        ("1. One", 1), ("first page text", 1), ("2. Two", 2), ("second page text", 2)]


# ── Re-ingest script ─────────────────────────────────────────────────────────

def test_reingest_rebuilds_pages_and_rechunks(monkeypatch):
    """Old one-chunk-per-page data (50-token overlap, no headings) becomes
    structure-aware chunks with page and heading metadata."""
    from data import qdrant_store
    from data.pipeline import ingestor
    from scripts import reingest
    from tests.fakes import FakeQdrant, fake_vector
    from qdrant_client.models import PointStruct

    qdrant = FakeQdrant()
    monkeypatch.setattr(qdrant_store, "_get_client", lambda: qdrant)
    monkeypatch.setattr(ingestor, "embed_texts", lambda texts: [fake_vector(t) for t in texts])

    page_texts = ["\n".join(lines) for lines in HR_PAGES]
    old = []
    for i, text in enumerate(page_texts):
        # Old chunker: each chunk began with the tail of the previous one.
        overlap = " ".join(page_texts[i - 1].split()[-8:]) + " " if i else ""
        old.append(PointStruct(id=i + 1, vector=fake_vector(text), payload={
            "document_id": "doc-hr", "document_title": "HR_Policy", "organization_id": "org-x",
            "collection_id": "col-hr", "collection_name": "HR", "page_number": i + 1,
            "chunk_index": i, "content": overlap + text,
        }))
    qdrant_store.upsert("org-x", old)

    assert reingest.main(["--org", "org-x"]) == 0
    new = qdrant.payloads(organization_id="org-x", document_id="doc-hr")
    assert len(new) > 3
    assert all(p["heading"] and p["page_number"] and p["document_version"] == 1 for p in new)
    assert all(p["collection_name"] == "HR" for p in new)
    leave = [p for p in new if p["heading"] == "8. Annual Leave"]
    assert leave and leave[0]["page_number"] == 3
    # No sentence was duplicated by leftover overlap.
    assert sum(p["content"].count("Standard working hours") for p in new) == 1

    # A second run skips the already re-chunked document.
    assert reingest.main(["--org", "org-x"]) == 0
    assert len(qdrant.payloads(organization_id="org-x", document_id="doc-hr")) == len(new)
