"""
data/pipeline/chunker.py
--------------------------
Structure-aware chunker (task C1).

Input is a list of blocks from processor.extract_text(): each block is
{"text", "page", "level"}, where level 0 is body text and level >= 1 is a
heading (1 = top level). Documents without structure fall back to
split_blocks(), which finds headings in plain text.

Rules:
  - Headings are hard boundaries: a chunk never spans two sections, so a
    citation always points at one section.
  - Each chunk is 400 to 700 tokens; a long section is split at sentence
    boundaries into evenly sized chunks with 80 tokens of overlap. A section
    shorter than 400 tokens is one (smaller) chunk.
  - Every chunk starts with its heading path, e.g. "8. Annual Leave > 8.1
    Entitlement", which is also stored as `heading`. Text before the first
    heading uses the document title.
  - Every chunk records the page it starts on (`page_number`) and ends on
    (`page_end`).

Token counts are estimated (1 token ≈ 0.75 words), as before.
"""

import math
import re

from agents.langgraph_agent.utils.utils import logger

MAX_TOKENS = 700
MIN_TOKENS = 400
OVERLAP_TOKENS = 80
# Long sentences are cut into windows of this size so the overlap tail and
# chunk sizes stay bounded.
MAX_UNIT_TOKENS = 120
WORDS_PER_TOKEN = 0.75
HEADING_SEPARATOR = " > "

# ── Heading detection for plain text (PDF, TXT, OCR output) ─────────────────

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
# "8. Annual Leave", "8.1 Entitlement", "2) Scope" — a number, then a capital.
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+([A-Z][^\n]*)$")
# Table-of-contents entries: dot leaders or a trailing page number.
_TOC_ENTRY = re.compile(r"(\.{3,}|…{2,}|\s{2,}|\t)\s*\d{1,4}\s*$|\s\d{1,4}$")
_MAX_HEADING_WORDS = 12


def heading_level(line: str) -> int:
    """Heading level of a single line of plain text, or 0 for body text."""
    line = line.strip()
    if not line or len(line) > 100:
        return 0
    if m := _MARKDOWN_HEADING.match(line):
        return len(m.group(1))
    if _TOC_ENTRY.search(line) or line[-1] in ".,;:!?":
        return 0
    words = line.split()
    if len(words) > _MAX_HEADING_WORDS:
        return 0
    if m := _NUMBERED_HEADING.match(line):
        return m.group(1).count(".") + 1
    letters = [c for c in line if c.isalpha()]
    if 2 <= len(words) <= 8 and len(letters) >= 4 and all(c.isupper() for c in letters) \
            and not any(c.isdigit() for c in line):
        return 1
    return 0


def _clean_heading(line: str) -> str:
    line = line.strip()
    if m := _MARKDOWN_HEADING.match(line):
        line = m.group(2)
    return re.sub(r"\s+", " ", line)


def split_blocks(text: str, page: int = 1) -> list[dict]:
    """Blocks from plain text: heading lines become heading blocks, other
    lines are grouped into paragraphs (a blank line or heading ends one).
    A form feed starts a new page."""
    blocks: list[dict] = []
    for page_offset, page_text in enumerate(text.split("\f")):
        paragraph: list[str] = []

        def flush():
            if paragraph:
                blocks.append({"text": " ".join(paragraph), "page": page + page_offset, "level": 0})
                paragraph.clear()

        for raw in page_text.splitlines():
            line = re.sub(r"[ \t]+", " ", raw).strip()
            if not line:
                flush()
                continue
            level = heading_level(line)
            if level:
                flush()
                blocks.append({"text": _clean_heading(line), "page": page + page_offset, "level": level})
            else:
                paragraph.append(line)
        flush()
    return blocks


# ── Chunking ─────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) / WORDS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _units(paragraphs: list[tuple[str, int]]) -> list[dict]:
    """Sentence-sized pieces of a section: {text, page, para, tokens}."""
    units = []
    for para_idx, (text, page) in enumerate(paragraphs):
        for sentence in _split_sentences(text):
            words = sentence.split()
            step = int(MAX_UNIT_TOKENS * WORDS_PER_TOKEN)
            for start in range(0, len(words), step):
                piece = " ".join(words[start:start + step])
                units.append({"text": piece, "page": page, "para": para_idx, "tokens": estimate_tokens(piece)})
    return units


def _join(units: list[dict]) -> str:
    out, last_para = "", None
    for u in units:
        if last_para is None:
            out = u["text"]
        else:
            out += (" " if u["para"] == last_para else "\n\n") + u["text"]
        last_para = u["para"]
    return out


def _overlap_tail(units: list[dict]) -> list[dict]:
    tail, tokens = [], 0
    for u in reversed(units):
        if tokens >= OVERLAP_TOKENS:
            break
        tail.insert(0, u)
        tokens += u["tokens"]
    return tail


def _chunk_section(heading: str, paragraphs: list[tuple[str, int]]) -> list[dict]:
    """Splits one section into chunks. Each chunk: (body, first page, last page)."""
    units = _units(paragraphs)
    if not units:
        return []
    prefix_tokens = estimate_tokens(heading) + 1
    budget = MAX_TOKENS - prefix_tokens
    total = sum(u["tokens"] for u in units)
    if total <= budget:
        return [{"body": _join(units), "page": units[0]["page"], "page_end": units[-1]["page"]}]

    # Even split: every chunk carries up to ~200 tokens of overlap, so new
    # content per chunk stays below budget minus that reserve.
    reserve = OVERLAP_TOKENS + MAX_UNIT_TOKENS
    n = math.ceil(total / max(budget - reserve, 1))
    target = total / n

    chunks, current, fresh_tokens, overlap = [], [], 0, []
    for u in units:
        size = sum(x["tokens"] for x in overlap + current)
        if current and (fresh_tokens >= target or size + u["tokens"] > budget):
            body_units = overlap + current
            chunks.append({"body": _join(body_units), "page": current[0]["page"], "page_end": current[-1]["page"]})
            overlap = _overlap_tail(body_units)
            current, fresh_tokens = [], 0
        current.append(u)
        fresh_tokens += u["tokens"]
    if current:
        body_units = overlap + current
        chunks.append({"body": _join(body_units), "page": current[0]["page"], "page_end": current[-1]["page"]})
    return chunks


def _sections(blocks: list[dict], title: str) -> list[tuple[str, list[tuple[str, int]]]]:
    """(heading path, [(paragraph, page)]) for every section with body text."""
    sections = []
    stack: list[tuple[int, str]] = []
    body: list[tuple[str, int]] = []

    def path() -> str:
        return HEADING_SEPARATOR.join(h for _, h in stack) or title

    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        level = block.get("level") or 0
        if level:
            if body:
                sections.append((path(), body))
                body = []
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            body.append((text, block.get("page") or 1))
    if body:
        sections.append((path(), body))
    return sections


def chunk_text(text: str, document_id: str, page_count: int = 1,
               blocks: list[dict] | None = None, title: str = "") -> list[dict]:
    """
    Split a document into chunks for embedding.

    Args:
        text:        full extracted text (used when `blocks` is not given)
        document_id: maps to documents.id
        page_count:  total pages (pages are clamped to it)
        blocks:      structured blocks from extract_text(), if available
        title:       document title, the heading for text before any heading

    Returns:
        [{chunk_index, content, token_count, page_number, page_end, heading,
          document_id}], content = heading path + blank line + body.
    """
    if blocks is None:
        if not text or not text.strip():
            logger.warning("Chunker → empty text, returning no chunks")
            return []
        blocks = split_blocks(text)

    title = (title or "").strip() or "Document"
    last_page = max(page_count or 1, 1)
    result = []
    for heading, paragraphs in _sections(blocks, title):
        for piece in _chunk_section(heading, paragraphs):
            content = f"{heading}\n\n{piece['body']}"
            result.append({
                "chunk_index": len(result),
                "content": content,
                "token_count": estimate_tokens(content),
                "page_number": min(piece["page"], last_page),
                "page_end": min(piece["page_end"], last_page),
                "heading": heading,
                "document_id": document_id,
            })

    if result:
        logger.info("Chunker → %d chunks (avg %d tokens) for %s", len(result),
                    sum(c["token_count"] for c in result) // len(result), document_id)
    else:
        logger.warning("Chunker → no body text found for %s", document_id)
    return result
