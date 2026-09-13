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
2. OCR fallback for PDFs — PyMuPDF renders the page to an image, RapidOCR
   reads it. Runs PER PAGE, only on pages pypdf found no text layer on, so a
   PDF that mixes digital and scanned pages keeps the (better) digital text
   and only pays the OCR cost for the scanned ones.
3. Docling fallback (if installed) — for complex layouts and tables

Scanned-page handling (the reason this file is shaped the way it is):
  pypdf returns "" for a page that is just a photo of text. The old code
  joined every page together and accepted the result whenever the WHOLE
  document had >100 chars — so a 20-page PDF with 2 digital and 18 scanned
  pages looked successful while silently dropping 18 pages of content. The
  agent would then answer "that is not in the documents" with full
  confidence. Detection is now per page, never per document.

Maps to ERD:
- documents.processing_status updated at each stage
- documents.page_count set after extraction
- documents.language detected and set
"""

import os
import tempfile
from pathlib import Path
from agents.langgraph_agent.utils.utils import logger

# A page with fewer than this many characters of extracted text is treated as
# having no text layer (i.e. scanned) and sent to OCR. Deliberately low —
# scanned pages usually yield 0-5 chars of junk, while even a sparse title
# page of real text clears 30.
OCR_MIN_CHARS_PER_PAGE = 30

# Render DPI for OCR. 200 is the sweet spot: 150 loses small print, 300 is
# ~2x slower for no accuracy gain on business documents.
OCR_RENDER_DPI = 200

# Hard cap so one pathological upload cannot hang the request thread.
OCR_MAX_PAGES = 100

_ocr_engine = None  # RapidOCR is expensive to construct — build once, reuse


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
    Extract PDF text page by page, OCR-ing only the pages that need it.

    Returns the usual dict plus:
      "pages"      — list[str], one entry per page, in order (lets the chunker
                     assign real page numbers instead of estimating them)
      "ocr_pages"  — how many pages went through OCR
    """
    pages: list[str] = []
    page_count = 0

    # ── Step 1: pypdf, per page ───────────────────────────────────────────
    try:
        from pypdf import PdfReader
        reader     = PdfReader(str(path))
        page_count = len(reader.pages)
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception as e:
                # One corrupt page must not sink the whole document — leave it
                # blank and let OCR try to recover it below.
                logger.warning("Processor → pypdf failed on a page (%s)", e)
                pages.append("")
    except Exception as e:
        logger.warning("Processor → pypdf could not open the PDF (%s)", e)
        pages, page_count = [], 0

    # ── Step 2: OCR the pages with no text layer ──────────────────────────
    ocr_count = 0
    if page_count:
        blank = [i for i, txt in enumerate(pages) if len(txt) < OCR_MIN_CHARS_PER_PAGE]

        if blank:
            logger.info(
                "Processor → %d of %d pages have no text layer, attempting OCR",
                len(blank), page_count
            )
            recovered = _ocr_pdf_pages(path, blank)
            for idx, txt in recovered.items():
                pages[idx] = txt
            ocr_count = len(recovered)

            still_blank = [
                i for i in blank
                if len(pages[i]) < OCR_MIN_CHARS_PER_PAGE
            ]
            if still_blank:
                # Loud on purpose: this is the failure that used to be silent.
                logger.warning(
                    "Processor → %d page(s) still have NO extractable text after OCR "
                    "(pages: %s). Their content is NOT in the index.",
                    len(still_blank),
                    ", ".join(str(i + 1) for i in still_blank[:20]),
                )

    text = "\n\n".join(p for p in pages if p).strip()

    # Accept if ANY page yielded real text. The old code required >100 chars
    # across the whole document, which threw away short one-page documents and
    # discarded perfectly good OCR output that happened to be brief. Emptiness
    # is now judged per page (above), so this gate only has to answer "did we
    # get anything at all".
    if any(len(p) >= OCR_MIN_CHARS_PER_PAGE for p in pages):
        method = "pypdf+ocr" if ocr_count else "pypdf"
        logger.info(
            "Processor → %s extracted %d chars from %d pages (%d via OCR)",
            method, len(text), page_count, ocr_count
        )
        return {
            "text": text,
            "pages": pages,
            "page_count": page_count,
            "ocr_pages": ocr_count,
            "method": method,
            "success": True,
        }

    # ── Step 3: Docling, last resort ──────────────────────────────────────
    logger.info("Processor → still no usable text, trying docling fallback")
    docling_result = _try_docling(path)
    if docling_result:
        return docling_result

    return {
        "success": False,
        "error": (
            "PDF text extraction failed — no text layer and OCR could not read it. "
            "Install OCR support with: pip install pymupdf rapidocr-onnxruntime"
        ),
    }


def _get_ocr_engine():
    """Build the RapidOCR engine once and cache it (model load is ~2-3s)."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        logger.info("Processor → loading RapidOCR engine (first use)")
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_pdf_pages(path: Path, page_indices: list[int]) -> dict:
    """
    OCR specific pages of a PDF.

    Args:
        path:         the PDF
        page_indices: 0-based page indices to read

    Returns:
        {page_index: text} for pages OCR actually recovered text from.
        Empty dict if OCR is unavailable — the caller logs and carries on with
        whatever digital text it has, rather than failing the whole upload.
    """
    if not page_indices:
        return {}

    if len(page_indices) > OCR_MAX_PAGES:
        logger.warning(
            "Processor → %d pages need OCR, capping at %d",
            len(page_indices), OCR_MAX_PAGES
        )
        page_indices = page_indices[:OCR_MAX_PAGES]

    # PyMuPDF renders; RapidOCR reads. Both are pure-pip (no Tesseract binary).
    try:
        try:
            import pymupdf as fitz          # PyMuPDF >= 1.24 preferred name
        except ImportError:
            import fitz                     # older releases
    except ImportError:
        logger.warning(
            "Processor → PyMuPDF not installed, cannot OCR scanned pages. "
            "Install with: pip install pymupdf rapidocr-onnxruntime"
        )
        return {}

    try:
        engine = _get_ocr_engine()
    except ImportError:
        logger.warning(
            "Processor → RapidOCR not installed, cannot OCR scanned pages. "
            "Install with: pip install pymupdf rapidocr-onnxruntime"
        )
        return {}
    except Exception as e:
        logger.error("Processor → could not start OCR engine: %s", e)
        return {}

    results: dict[int, str] = {}
    doc = None
    try:
        doc = fitz.open(str(path))
        for idx in page_indices:
            tmp_png = None
            try:
                pixmap = doc[idx].get_pixmap(dpi=OCR_RENDER_DPI)

                # RapidOCR takes a path; write the render to a temp PNG and
                # clean it up immediately so nothing accumulates on disk.
                fd, tmp_png = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pixmap.save(tmp_png)

                ocr_result, _elapsed = engine(tmp_png)
                if not ocr_result:
                    logger.info("Processor → OCR found no text on page %d", idx + 1)
                    continue

                # ocr_result rows are [bounding_box, text, confidence]
                page_text = "\n".join(
                    row[1].strip() for row in ocr_result
                    if len(row) > 1 and row[1] and row[1].strip()
                ).strip()

                if page_text:
                    results[idx] = page_text
                    logger.info(
                        "Processor → OCR read page %d (%d chars)",
                        idx + 1, len(page_text)
                    )

            except Exception as e:
                logger.warning("Processor → OCR failed on page %d: %s", idx + 1, e)
            finally:
                if tmp_png and os.path.exists(tmp_png):
                    try:
                        os.remove(tmp_png)
                    except OSError:
                        pass

    except Exception as e:
        logger.error("Processor → OCR pass failed: %s", e)
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    return results


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