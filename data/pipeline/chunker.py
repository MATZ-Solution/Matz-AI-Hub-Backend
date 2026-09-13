"""
data/pipeline/chunker.py
--------------------------
Text chunker — splits extracted text into overlapping chunks.

Why overlapping chunks?
  If a sentence spans the boundary of two chunks, overlapping ensures
  neither chunk loses the context. Example:

  Chunk 1: "...passwords must be 12 characters. MFA is mandatory..."
  Chunk 2: "...MFA is mandatory for all systems. Never share..."
             ↑ overlap keeps context from chunk 1

Settings:
  CHUNK_SIZE    = 500 tokens (~375 words) — good for RAG
  CHUNK_OVERLAP = 50 tokens  — enough context overlap

Maps to ERD:
  - document_chunks.content      → chunk text
  - document_chunks.chunk_index  → position in document
  - document_chunks.page_number  → real page number when per-page text is
                                   available (PDFs), estimated otherwise
  - document_chunks.token_count  → approximate token count
"""

import re
from agents.langgraph_agent.utils.utils import logger

CHUNK_SIZE    = 500   # target tokens per chunk
CHUNK_OVERLAP = 50    # overlap tokens between chunks
WORDS_PER_TOKEN = 0.75  # rough estimate: 1 token ≈ 0.75 words


def chunk_text(
    text: str,
    document_id: str,
    page_count: int = 1,
    pages: list[str] | None = None,
) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text:        full extracted text (ignored when `pages` is given)
        document_id: maps to documents.id in ERD
        page_count:  total pages in document (for page estimation)
        pages:       per-page text, in order. The PDF extractor supplies this,
                     which lets each chunk carry its REAL page number. Without
                     it we fall back to estimating the page from the chunk's
                     relative position, which drifts badly on documents with
                     uneven page lengths (a dense page 1 and a sparse page 9
                     are treated as the same size).

    Returns:
        list of chunk dicts ready for embedding and Qdrant upload
    """
    # Page boundaries as (start, end) character offsets into the joined text,
    # so a chunk's offset can be mapped back to the page it came from.
    bounds: list[tuple[int, int]] | None = None

    if pages:
        cleaned = [_clean_text(p) for p in pages]
        parts, bounds, offset = [], [], 0
        for page_text in cleaned:
            parts.append(page_text)
            bounds.append((offset, offset + len(page_text)))
            offset += len(page_text) + 2          # the "\n\n" join below
        text = "\n\n".join(parts)
        page_count = len(cleaned)
    else:
        text = _clean_text(text)

    if not text or not text.strip():
        logger.warning("Chunker → empty text, returning no chunks")
        return []

    # Split into sentences first — don't cut mid-sentence
    sentences = _split_sentences(text)

    # Group sentences into chunks
    chunks = _group_into_chunks(sentences)

    # Build chunk dicts with ERD fields
    result = []
    total_chars = len(text)

    # Chunks overlap and can repeat text, so a plain text.find() would keep
    # matching the FIRST occurrence and pin every repeat to the wrong page.
    # Advance a cursor instead so each chunk is located at or after the
    # previous one.
    search_from = 0

    for i, chunk_text in enumerate(chunks):
        probe          = chunk_text[:50]
        chunk_position = text.find(probe, search_from)
        if chunk_position < 0:                      # overlap rewound past the cursor
            chunk_position = text.find(probe)
        if chunk_position < 0:
            chunk_position = search_from
        search_from = max(search_from, chunk_position + 1)

        if bounds:
            page_number = _page_for_offset(chunk_position, bounds)
        else:
            page_number = max(1, int((chunk_position / max(total_chars, 1)) * page_count) + 1)

        result.append({
            "chunk_index":  i,
            "content":      chunk_text,
            "token_count":  _estimate_tokens(chunk_text),
            "page_number":  page_number,
            "document_id":  document_id,
        })

    logger.info(
        "Chunker → %d chunks from %d chars (avg %d tokens/chunk)",
        len(result),
        len(text),
        sum(c["token_count"] for c in result) // max(len(result), 1)
    )
    return result


def _page_for_offset(offset: int, bounds: list[tuple[int, int]]) -> int:
    """Map a character offset in the joined text back to a 1-based page number."""
    for i, (start, end) in enumerate(bounds):
        if start <= offset <= end:
            return i + 1
    # Past the last boundary (can happen on the final chunk) → last page.
    return len(bounds) or 1


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 newlines
    text = re.sub(r' {2,}', ' ', text)        # max 1 space
    text = text.strip()
    return text


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Also split on paragraph breaks
    result = []
    for sentence in sentences:
        if '\n\n' in sentence:
            parts = sentence.split('\n\n')
            result.extend([p.strip() for p in parts if p.strip()])
        else:
            if sentence.strip():
                result.append(sentence.strip())
    return result


def _group_into_chunks(sentences: list[str]) -> list[str]:
    """
    Group sentences into chunks of approximately CHUNK_SIZE tokens.
    Uses overlap of CHUNK_OVERLAP tokens between chunks.
    """
    chunks   = []
    current  = []
    cur_tokens = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence exceeds chunk size → finalize current chunk
        if cur_tokens + sentence_tokens > CHUNK_SIZE and current:
            chunks.append(" ".join(current))

            # Keep last few sentences for overlap
            overlap_tokens = 0
            overlap_sents  = []
            for s in reversed(current):
                overlap_tokens += _estimate_tokens(s)
                overlap_sents.insert(0, s)
                if overlap_tokens >= CHUNK_OVERLAP:
                    break

            current    = overlap_sents
            cur_tokens = overlap_tokens

        current.append(sentence)
        cur_tokens += sentence_tokens

    # Add the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words / 0.75."""
    word_count = len(text.split())
    return int(word_count / WORDS_PER_TOKEN)