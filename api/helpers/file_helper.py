"""
api/helpers/file_helper.py
-----------------------------
Upload validation and temp storage, shared by /ingest and PUT /documents/{id}.

Server-side checks (task A5), in order:
  1. extension allow-list (pdf, docx, pptx, xlsx, txt)
  2. size: not empty, at most 50 MB (counted while copying, so an oversized
     upload is never written to disk in full)
  3. magic bytes: the content must really be the type its extension claims
  4. not encrypted or password-protected
Every rejection is an `invalid_file` error with a sentence the user can act on.
"""

import os
import re
import tempfile
import unicodedata
import zipfile

from fastapi import UploadFile

from api.config.settings import ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_BYTES
from api.errors import ApiError, ErrorCode

_COPY_CHUNK = 1024 * 1024

# OOXML formats are ZIP archives; each must contain its main part.
_OOXML_MAIN_PART = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
    ".xlsx": "xl/workbook.xml",
}
_ZIP_MAGIC = b"PK\x03\x04"
_PDF_MAGIC = b"%PDF-"
# Password-protected Office files are not ZIPs but OLE compound files.
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_UNSUPPORTED_TYPE = "This file type isn't supported. Use PDF, DOCX, PPTX, XLSX or TXT."
_PROTECTED = "This file is password-protected. Remove the password and upload it again."


def _invalid(message: str, detail: str | None = None) -> ApiError:
    return ApiError(ErrorCode.invalid_file, message, log_detail=detail)


def sanitize_filename(filename: str | None) -> str:
    """A safe display/log name: no directories, control characters or odd
    symbols, at most 120 characters, extension kept."""
    name = unicodedata.normalize("NFKC", filename or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w .()\-]", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        return "document"
    stem, dot, ext = name.rpartition(".")
    if dot and len(ext) <= 10:
        return stem[: 120 - len(ext) - 1] + "." + ext
    return name[:120]


def get_file_extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ""


def is_allowed_extension(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_UPLOAD_EXTENSIONS


def _copy_with_limit(file: UploadFile, suffix: str) -> tuple[str, int]:
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            while chunk := file.file.read(_COPY_CHUNK):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise _invalid(f"This file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Upload a smaller file.")
                tmp.write(chunk)
        except Exception:
            tmp.close()
            os.remove(tmp.name)
            raise
        return tmp.name, size


def _check_pdf(path: str, head: bytes) -> None:
    if not head.startswith(_PDF_MAGIC):
        raise _invalid("This file isn't a valid PDF.", "missing %PDF- header")
    from pypdf import PdfReader
    try:
        reader = PdfReader(path)
        # Owner-restricted PDFs open with an empty password and are readable;
        # only a PDF that needs a password to open is refused.
        if reader.is_encrypted and not reader.decrypt(""):
            raise _invalid(_PROTECTED, "encrypted pdf")
    except ApiError:
        raise
    except Exception as e:
        raise _invalid("This PDF is damaged and can't be read.", f"pypdf: {e}")


def _check_ooxml(path: str, head: bytes, ext: str) -> None:
    if head.startswith(_OLE_MAGIC):
        raise _invalid(_PROTECTED, "OLE container (encrypted office file)")
    if not head.startswith(_ZIP_MAGIC):
        raise _invalid(f"This file isn't a valid {ext[1:].upper()} document.", "missing zip header")
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except zipfile.BadZipFile as e:
        raise _invalid(f"This {ext[1:].upper()} file is damaged and can't be read.", f"zip: {e}")
    if _OOXML_MAIN_PART[ext] not in names:
        raise _invalid(f"This file isn't a valid {ext[1:].upper()} document.", "main part missing")


def _check_txt(path: str) -> None:
    with open(path, "rb") as f:
        sample = f.read(64 * 1024)
    if b"\x00" in sample:
        raise _invalid("This isn't a plain text file.", "NUL byte in .txt")
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as e:
        # A multi-byte character cut at the sample boundary is fine.
        if e.start < len(sample) - 4:
            raise _invalid("This text file isn't UTF-8 encoded. Save it as UTF-8 and upload it again.", str(e))


def validate_and_save_upload(file: UploadFile) -> tuple[str, str]:
    """Runs every upload check and returns (temp path, sanitized filename).
    The caller must pass the path to cleanup_tempfile() when done."""
    safe_name = sanitize_filename(file.filename)
    ext = get_file_extension(safe_name)
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise _invalid(_UNSUPPORTED_TYPE, f"extension {ext!r}")

    path, size = _copy_with_limit(file, ext)
    try:
        if size == 0:
            raise _invalid("This file is empty.")
        with open(path, "rb") as f:
            head = f.read(8)
        if ext == ".pdf":
            _check_pdf(path, head)
        elif ext in _OOXML_MAIN_PART:
            _check_ooxml(path, head, ext)
        else:
            _check_txt(path)
    except Exception:
        cleanup_tempfile(path)
        raise
    return path, safe_name


def cleanup_tempfile(tmp_path: str) -> None:
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)
