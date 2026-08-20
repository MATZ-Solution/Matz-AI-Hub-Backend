"""
data/pipeline/processor.py
---------------------------
Document processor — extracts text from uploaded files.

Strategy:
1. Try PyPDF first (fast, no ML, works for digital PDFs)
2. If PyPDF returns empty → use Docling (handles scanned PDFs, tables, images)

Supported formats:
- PDF  → PyPDF first, Docling fallback
- DOCX → Docling
- PPTX → Docling
- XLSX → Docling
- TXT  → direct read

Maps to ERD:
- documents.processing_status updated at each stage
- documents.page_count set after extraction
- documents.language detected and set
"""

import os
from pathlib import Path
from agent.src.utils.logger import logger


def extract_text(file_path: str) -> dict:
    """
    Extract text from a document file.

    Args:
        file_path: path to the document file

    Returns:
        {
            "text": full extracted text,
            "page_count": number of pages,
            "method": "pypdf" | "docling" | "plain",
            "success": True | False,
            "error": error message if failed
        }
    """
    path = Path(file_path)

    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    ext = path.suffix.lower()
    logger.info("Processor → extracting text from %s (%s)", path.name, ext)

    if ext == ".txt":
        return _extract_txt(path)
    elif ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".docx", ".pptx", ".xlsx"):
        return _extract_with_docling(path)
    else:
        return {"success": False, "error": f"Unsupported file type: {ext}"}


def _extract_txt(path: Path) -> dict:
    """Plain text extraction."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        logger.info("Processor → plain text extracted (%d chars)", len(text))
        return {
            "text": text,
            "page_count": 1,
            "method": "plain",
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _extract_pdf(path: Path) -> dict:
    """
    Try PyPDF first — fast, no ML needed.
    Fall back to Docling if PyPDF returns empty (scanned PDF).
    """
    # Step 1: Try PyPDF
    try:
        from pypdf import PdfReader
        reader     = PdfReader(str(path))
        page_count = len(reader.pages)
        text       = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if text and len(text) > 100:
            logger.info(
                "Processor → PyPDF extracted %d chars from %d pages",
                len(text), page_count
            )
            return {
                "text": text,
                "page_count": page_count,
                "method": "pypdf",
                "success": True,
            }
        else:
            logger.info("Processor → PyPDF returned empty text, falling back to Docling")

    except Exception as e:
        logger.warning("Processor → PyPDF failed (%s), falling back to Docling", e)

    # Step 2: Docling fallback (scanned PDFs, tables, images)
    return _extract_with_docling(path)


def _extract_with_docling(path: Path) -> dict:
    """
    Use Docling for scanned PDFs, DOCX, PPTX, XLSX.
    Docling handles OCR, tables, and complex layouts.
    Output is clean Markdown text.
    """
    try:
        from docling.document_converter import DocumentConverter
        logger.info("Processor → running Docling on %s", path.name)

        converter = DocumentConverter()
        result    = converter.convert(str(path))
        text      = result.document.export_to_markdown()
        page_count = getattr(result.document, 'page_count', 1) or 1

        if not text or len(text) < 50:
            return {"success": False, "error": "Docling extracted empty text"}

        logger.info(
            "Processor → Docling extracted %d chars from %d pages",
            len(text), page_count
        )
        return {
            "text": text,
            "page_count": page_count,
            "method": "docling",
            "success": True,
        }

    except Exception as e:
        logger.error("Processor → Docling failed: %s", e)
        return {"success": False, "error": str(e)}