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
2. Vision-model fallback for scanned PDF pages — PyMuPDF renders the page to
   an image and a multimodal LLM (Fireworks / DeepSeek) transcribes it. Runs
   PER PAGE, only on pages with no text layer, so a PDF mixing digital and
   scanned pages keeps the (better, free) digital text and only sends the
   scanned pages to the API.
3. Docling fallback (if installed) — for complex layouts and tables

Maps to ERD:
- documents.processing_status updated at each stage
- documents.page_count set after extraction
- documents.language detected and set
"""

import os
import re
from pathlib import Path
from agents.langgraph_agent.utils.utils import logger

# A page yielding fewer than this many characters is treated as having no text
# layer — i.e. it is scanned (an image of text) rather than digital. Scanned
# pages typically return 0-5 characters of noise; even a sparse title page of
# real text clears 30 comfortably.
MIN_CHARS_PER_PAGE = 30

# ── Vision-model transcription of scanned pages ───────────────────────────
# Fireworks is OpenAI-API-compatible, so this is a plain chat-completions call
# with an image part. Set FIREWORKS_API_KEY in .env to enable; with no key the
# feature stays off and scanned pages are simply reported as unreadable.
VISION_API_URL   = "https://api.fireworks.ai/inference/v1/chat/completions"
VISION_MODEL     = os.environ.get(
    "FIREWORKS_VISION_MODEL",
    "accounts/fireworks/models/deepseek-v4-flash-vision-exp",
)
VISION_RENDER_DPI = 200    # 150 loses small print; 300 costs more for no gain
VISION_MAX_PAGES  = 50     # cost guard — one upload cannot run away
VISION_TIMEOUT    = 120    # seconds per page

# Transcription only. Without this the model tends to summarise or comment,
# which would poison the index with text that is not in the document.
VISION_PROMPT = (
    "Transcribe ALL text visible in this document page, exactly as written. "
    "Preserve the reading order, headings, and list or table structure. "
    "Output ONLY the transcribed text — no preamble, no commentary, no "
    "markdown code fences, and do not describe the image. If the page has no "
    "readable text, output nothing at all."
)


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
    Extract PDF text page by page.

    Emptiness is judged PER PAGE, never per document. The previous version
    joined every page together and accepted the result whenever the whole
    document held more than 100 characters, so a 20-page PDF with 2 digital
    and 18 scanned pages returned success while silently dropping 18 pages of
    content — and the assistant would then answer "that is not in the
    documents" with complete confidence. Pages with no text layer are now
    identified and reported.

    Returns the usual dict plus:
      "pages"         — list[str], one entry per page, in order
      "scanned_pages" — 1-based page numbers that had no text layer
    """
    pages: list[str] = []
    page_count = 0

    try:
        from pypdf import PdfReader
        reader     = PdfReader(str(path))
        page_count = len(reader.pages)
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception as e:
                # One bad page must not sink the document — record it as empty
                # so it is reported below rather than silently disappearing.
                logger.warning("Processor → pypdf failed on a page (%s)", e)
                pages.append("")
    except Exception as e:
        logger.warning("Processor → pypdf could not open the PDF (%s)", e)
        pages, page_count = [], 0

    scanned = [i + 1 for i, txt in enumerate(pages) if len(txt) < MIN_CHARS_PER_PAGE]

    # ── Transcribe scanned pages with the vision model ────────────────────
    vision_count = 0
    if scanned:
        recovered = _vision_read_pages(path, [n - 1 for n in scanned])
        for idx, txt in recovered.items():
            pages[idx] = txt
        vision_count = len(recovered)
        # Anything still empty stays in `scanned` and is reported below.
        scanned = [
            i + 1 for i, txt in enumerate(pages) if len(txt) < MIN_CHARS_PER_PAGE
        ]

    if scanned:
        logger.warning(
            "Processor → %d of %d page(s) have NO text layer and were NOT "
            "extracted (pages: %s). This PDF is scanned or partly scanned — "
            "that content will be missing from the index and the assistant "
            "will not be able to answer questions about it.",
            len(scanned), page_count,
            ", ".join(str(n) for n in scanned[:20]) + ("..." if len(scanned) > 20 else ""),
        )

    text = "\n\n".join(p for p in pages if p).strip()

    # Accept if ANY page yielded real text. The old >100-chars-per-document
    # rule also rejected legitimately short one-page documents outright.
    if any(len(p) >= MIN_CHARS_PER_PAGE for p in pages):
        method = "pypdf+vision" if vision_count else "pypdf"
        logger.info(
            "Processor → %s extracted %d chars from %d of %d pages (%d via vision model)",
            method, len(text), page_count - len(scanned), page_count, vision_count
        )
        return {
            "text": text,
            "pages": pages,
            "page_count": page_count,
            "scanned_pages": scanned,
            "vision_pages": vision_count,
            "method": method,
            "success": True,
        }

    logger.info("Processor → no text layer anywhere, trying docling fallback")
    docling_result = _try_docling(path)
    if docling_result:
        return docling_result

    return {
        "success": False,
        "error": (
            "PDF text extraction failed — this document has no text layer on any "
            "page (it is a scan or photo of text) and the vision model could not "
            "read it. Check that FIREWORKS_API_KEY is set."
        ),
    }


def _render_page_png(doc, index: int) -> bytes:
    """Render one PDF page to PNG bytes."""
    return doc[index].get_pixmap(dpi=VISION_RENDER_DPI).tobytes("png")


def _vision_read_pages(path: Path, page_indices: list[int]) -> dict:
    """
    Transcribe scanned PDF pages with a multimodal LLM.

    Args:
        path:         the PDF
        page_indices: 0-based indices of pages with no text layer

    Returns:
        {page_index: text} for pages the model actually read. Returns {} when
        the feature is unavailable (no API key, PyMuPDF missing, API down) —
        the caller keeps whatever digital text it has and reports the rest as
        unreadable, rather than failing the whole upload.
    """
    if not page_indices:
        return {}

    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "Processor → FIREWORKS_API_KEY not set, cannot read %d scanned page(s)",
            len(page_indices),
        )
        return {}

    if len(page_indices) > VISION_MAX_PAGES:
        logger.warning(
            "Processor → %d pages need transcription, capping at %d to limit cost",
            len(page_indices), VISION_MAX_PAGES,
        )
        page_indices = page_indices[:VISION_MAX_PAGES]

    try:
        try:
            import pymupdf as fitz          # PyMuPDF >= 1.24 preferred name
        except ImportError:
            import fitz                     # older releases
    except ImportError:
        logger.warning(
            "Processor → PyMuPDF not installed, cannot render scanned pages. "
            "Install with: pip install pymupdf"
        )
        return {}

    import base64
    import httpx

    results: dict[int, str] = {}
    doc = None
    try:
        doc = fitz.open(str(path))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=VISION_TIMEOUT) as client:
            for idx in page_indices:
                try:
                    png_b64 = base64.b64encode(_render_page_png(doc, idx)).decode()
                    payload = {
                        "model": VISION_MODEL,
                        "temperature": 0,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VISION_PROMPT},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{png_b64}"
                                }},
                            ],
                        }],
                    }

                    response = client.post(VISION_API_URL, headers=headers, json=payload)
                    if response.status_code != 200:
                        # Body carries the real reason (bad key, quota, bad model id).
                        logger.error(
                            "Processor → vision API returned %s on page %d: %s",
                            response.status_code, idx + 1, response.text[:300],
                        )
                        continue

                    data   = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    page_text = (choice.get("message", {}).get("content") or "").strip()

                    # Strip a stray code fence if the model adds one anyway.
                    if page_text.startswith("```"):
                        page_text = re.sub(r"^```[a-zA-Z]*\s*", "", page_text)
                        page_text = re.sub(r"\s*```$", "", page_text).strip()

                    if choice.get("finish_reason") == "length":
                        logger.warning(
                            "Processor → vision output TRUNCATED on page %d — "
                            "the transcription is incomplete", idx + 1,
                        )

                    if page_text:
                        results[idx] = page_text
                        logger.info(
                            "Processor → vision model read page %d (%d chars)",
                            idx + 1, len(page_text),
                        )
                    else:
                        logger.info(
                            "Processor → vision model found no text on page %d", idx + 1
                        )

                except Exception as e:
                    logger.warning(
                        "Processor → vision transcription failed on page %d: %s",
                        idx + 1, e,
                    )

    except Exception as e:
        logger.error("Processor → vision transcription pass failed: %s", e)
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