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
  - document_chunks.page_number  → estimated page number
  - document_chunks.token_count  → approximate token count
"""

import re
from agent.src.utils.logger import logger

CHUNK_SIZE    = 500   # target tokens per chunk
CHUNK_OVERLAP = 50    # overlap tokens between chunks
WORDS_PER_TOKEN = 0.75  # rough estimate: 1 token ≈ 0.75 words


def chunk_text(text: str, document_id: str, page_count: int = 1) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text:        full extracted text
        document_id: maps to documents.id in ERD
        page_count:  total pages in document (for page estimation)

    Returns:
        list of chunk dicts ready for embedding and Qdrant upload
    """
    if not text or not text.strip():
        logger.warning("Chunker → empty text, returning no chunks")
        return []

    # Clean text
    text = _clean_text(text)

    # Split into sentences first — don't cut mid-sentence
    sentences = _split_sentences(text)

    # Group sentences into chunks
    chunks = _group_into_chunks(sentences)

    # Build chunk dicts with ERD fields
    result = []
    total_chars = len(text)

    for i, chunk_text in enumerate(chunks):
        # Estimate page number based on position in document
        chunk_position = text.find(chunk_text[:50])
        estimated_page = max(1, int((chunk_position / max(total_chars, 1)) * page_count) + 1)

        result.append({
            "chunk_index":  i,
            "content":      chunk_text,
            "token_count":  _estimate_tokens(chunk_text),
            "page_number":  estimated_page,
            "document_id":  document_id,
        })

    logger.info(
        "Chunker → %d chunks from %d chars (avg %d tokens/chunk)",
        len(result),
        len(text),
        sum(c["token_count"] for c in result) // max(len(result), 1)
    )
    return result


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