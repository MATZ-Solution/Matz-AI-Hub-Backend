"""
scripts/reingest.py — re-chunk existing documents with the C1 chunker.

Original files are not stored (only their chunks in Qdrant), so each
document is rebuilt from its current chunks: chunks are ordered by
chunk_index, the overlap between neighbours is removed, the text is grouped
back into pages, and the result goes through the structure-aware chunker
and embedding again. The old chunks are deleted and the new ones uploaded
under the same document id, collection and version.

Documents already chunked by C1 (their chunks carry `heading`) are skipped
unless --force is given.

Run from Backend/ (needs QDRANT_URL, Supabase and the embedding model):
    .venv/bin/python -m scripts.reingest --org matz-demo-org --dry-run
    .venv/bin/python -m scripts.reingest --org matz-demo-org
"""

import argparse
import re
import sys
from collections import defaultdict

from data import qdrant_store
from data.pipeline.chunker import chunk_text, split_blocks
from data.pipeline.ingestor import delete_document_chunks, ingest_chunks
from api.config.settings import DEFAULT_ORGANIZATION_ID


def _strip_overlap(previous: str, current: str) -> str:
    """`current` without the words it repeats from the end of `previous`,
    keeping the rest of its line breaks (they mark headings)."""
    a = previous.split()
    words = list(re.finditer(r"\S+", current))
    b = [m.group() for m in words]
    for n in range(min(len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            return current[words[n - 1].end():].lstrip(" \t").lstrip("\n")
    return current


def rebuild_pages(payloads: list[dict]) -> list[tuple[int, str]]:
    """[(page, text)] in page order, from one document's chunk payloads."""
    ordered = sorted(payloads, key=lambda p: p.get("chunk_index", 0))
    pages: dict[int, list[str]] = defaultdict(list)
    previous = ""
    for p in ordered:
        content = p.get("content", "")
        if p.get("heading") and content.startswith(p["heading"] + "\n\n"):
            content = content[len(p["heading"]) + 2:]
        text = _strip_overlap(previous, content) if previous else content
        previous = content
        if text.strip():
            pages[int(p.get("page_number") or 1)].append(text)
    return [(page, "\n\n".join(parts)) for page, parts in sorted(pages.items())]


def reingest_document(org: str, payloads: list[dict], dry_run: bool) -> int:
    first = payloads[0]
    doc_id = first["document_id"]
    pages = rebuild_pages(payloads)
    blocks = [b for page, text in pages for b in split_blocks(text, page)]
    page_count = max((p for p, _ in pages), default=1)
    chunks = chunk_text("", doc_id, page_count, blocks=blocks, title=first.get("document_title", ""))
    print(f"  {first.get('document_title', doc_id)}: {len(payloads)} old chunks → {len(chunks)} new"
          f" over {page_count} page(s)")
    if dry_run or not chunks:
        return len(chunks)
    delete_document_chunks(doc_id, org)
    ingest_chunks(
        chunks=chunks, document_id=doc_id, document_title=first.get("document_title", ""),
        collection_id=first.get("collection_id", ""), collection_name=first.get("collection_name", ""),
        organization_id=org, document_version=int(first.get("document_version") or 1),
    )
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--org", default=DEFAULT_ORGANIZATION_ID, help="organization to re-ingest")
    parser.add_argument("--document", help="only this document id")
    parser.add_argument("--force", action="store_true", help="also re-chunk documents already chunked by C1")
    parser.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = parser.parse_args(argv)

    conditions = [qdrant_store.match("document_id", args.document)] if args.document else []
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for point in qdrant_store.scroll_all(args.org, *conditions):
        by_doc[point.payload.get("document_id", "unknown")].append(point.payload)

    print(f"Organization {args.org}: {len(by_doc)} document(s){' (dry run)' if args.dry_run else ''}")
    done = 0
    for doc_id, payloads in by_doc.items():
        if not args.force and all(p.get("heading") for p in payloads):
            print(f"  {payloads[0].get('document_title', doc_id)}: already structure-aware, skipped")
            continue
        reingest_document(args.org, payloads, args.dry_run)
        done += 1
    print(f"Re-ingested {done} document(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
