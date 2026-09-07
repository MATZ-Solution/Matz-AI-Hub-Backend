"""
data/pipeline/processor.py
---------------------------
Document processor — extracts text from uploaded files.

Strategy:
1. Lightweight extractors (no ML, fast) — primary:
   - PDF  → pypdf
   - DOCX → python-docx
   - PPTX → python-pptx
   - XLSX → openpyxl
   - TXT  → direct read
2. Docling fallback (if installed) — for complex layouts, scanned PDFs, OCR

Maps to ERD:
- documents.processing_status updated at each stage
- documents.page_count set after extraction
- documents.language detected and set
"""

import os
from pathlib import Path
from agents.langgraph_agent.utils.utils import logger


def extract_text(file_path: str) -> dict:
    """
    Extract text from a document file.

    Args:
        file_path: path to the document file

    Returns:
        {
            "text": full extracted text,
            "page_count": number of pages,
            "method": "pypdf" | "python-docx" | "python-pptx" | "openpyxl" | "docling" | "plain",
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
    elif ext == ".docx":
        return _extract_docx(path)
    elif ext == ".pptx":
        return _extract_pptx(path)
    elif ext == ".xlsx":
        return _extract_xlsx(path)
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
    Try pypdf first — fast, no ML needed.
    Fall back to Docling if pypdf returns empty (scanned PDF) AND docling is available.
    """
    # Step 1: Try pypdf
    try:
        from pypdf import PdfReader
        reader     = PdfReader(str(path))
        page_count = len(reader.pages)
        text       = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if text and len(text) > 100:
            logger.info(
                "Processor → pypdf extracted %d chars from %d pages",
                len(text), page_count
            )
            return {
                "text": text,
                "page_count": page_count,
                "method": "pypdf",
                "success": True,
            }
        else:
            logger.info("Processor → pypdf returned empty text, trying docling fallback")

    except Exception as e:
        logger.warning("Processor → pypdf failed (%s), trying docling fallback", e)

    # Step 2: Docling fallback (scanned PDFs — optional, if installed)
    docling_result = _try_docling(path)
    if docling_result:
        return docling_result

    return {"success": False, "error": "PDF text extraction failed (pypdf returned empty, docling not available for OCR)"}


def _extract_docx(path: Path) -> dict:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(str(path))

        lines: list[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                lines.append(para.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    lines.append(row_text)

        text = "\n\n".join(lines)
        if not text or len(text) < 50:
            docling_result = _try_docling(path)
            if docling_result:
                return docling_result
            return {"success": False, "error": "DOCX extraction returned empty text"}

        page_count = max(1, len(doc.paragraphs) // 50)
        logger.info("Processor → python-docx extracted %d chars (~%d pages)", len(text), page_count)
        return {
            "text": text,
            "page_count": page_count,
            "method": "python-docx",
            "success": True,
        }
    except Exception as e:
        logger.warning("Processor → python-docx failed (%s), trying docling fallback", e)
        docling_result = _try_docling(path)
        if docling_result:
            return docling_result
        return {"success": False, "error": f"DOCX extraction failed: {e}"}


def _extract_pptx(path: Path) -> dict:
    """Extract text from PPTX using python-pptx."""
    try:
        from pptx import Presentation
        prs = Presentation(str(path))

        slides_text: list[str] = []
        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_lines: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_lines.append(shape.text.strip())
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            slide_lines.append(row_text)
            if slide_lines:
                slides_text.append(f"--- Slide {slide_idx} ---\n" + "\n".join(slide_lines))

        text = "\n\n".join(slides_text)
        page_count = len(prs.slides) or 1

        if not text or len(text) < 50:
            docling_result = _try_docling(path)
            if docling_result:
                return docling_result
            return {"success": False, "error": "PPTX extraction returned empty text"}

        logger.info("Processor → python-pptx extracted %d chars from %d slides", len(text), page_count)
        return {
            "text": text,
            "page_count": page_count,
            "method": "python-pptx",
            "success": True,
        }
    except Exception as e:
        logger.warning("Processor → python-pptx failed (%s), trying docling fallback", e)
        docling_result = _try_docling(path)
        if docling_result:
            return docling_result
        return {"success": False, "error": f"PPTX extraction failed: {e}"}


def _extract_xlsx(path: Path) -> dict:
    """Extract text from XLSX using openpyxl."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(path), data_only=True, read_only=True)

        sheets_text: list[str] = []
        total_rows = 0

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_text: list[str] = []
            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join(str(cell).strip() for cell in row if cell is not None and str(cell).strip())
                if row_text:
                    rows_text.append(row_text)
                total_rows += 1
            if rows_text:
                sheets_text.append(f"### Sheet: {sheet_name}\n" + "\n".join(rows_text))

        text = "\n\n".join(sheets_text)
        page_count = max(1, total_rows // 60)

        if not text or len(text) < 50:
            docling_result = _try_docling(path)
            if docling_result:
                return docling_result
            return {"success": False, "error": "XLSX extraction returned empty text"}

        logger.info("Processor → openpyxl extracted %d chars (%d rows, ~%d pages)", len(text), total_rows, page_count)
        return {
            "text": text,
            "page_count": page_count,
            "method": "openpyxl",
            "success": True,
        }
    except Exception as e:
        logger.warning("Processor → openpyxl failed (%s), trying docling fallback", e)
        docling_result = _try_docling(path)
        if docling_result:
            return docling_result
        return {"success": False, "error": f"XLSX extraction failed: {e}"}


def _try_docling(path: Path) -> dict | None:
    """
    Try Docling for complex layouts, OCR, tables.
    Returns None if docling is not installed or fails.
    """
    try:
        from docling.document_converter import DocumentConverter
        logger.info("Processor → running Docling on %s", path.name)

        converter = DocumentConverter()
        result    = converter.convert(str(path))
        text      = result.document.export_to_markdown()
        page_count = getattr(result.document, 'page_count', 1) or 1

        if not text or len(text) < 50:
            logger.warning("Processor → Docling extracted empty text")
            return None

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

    except ImportError:
        logger.info("Processor → Docling not installed, skipping optional fallback")
        return None
    except Exception as e:
        logger.error("Processor → Docling failed: %s", e)
        return None