"""
api/helpers/file_helper.py
-----------------------------
Small helpers for handling uploaded files (validation + temp storage)
shared by the knowledge controller's /ingest and /documents/{id} (PUT) flows.
"""

import os
import shutil
import tempfile

from fastapi import UploadFile

from api.config.settings import ALLOWED_UPLOAD_EXTENSIONS


def get_file_extension(filename: str) -> str:
    return "." + filename.split(".")[-1].lower() if filename else ""


def is_allowed_extension(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_UPLOAD_EXTENSIONS


def save_upload_to_tempfile(file: UploadFile) -> str:
    """Copies an UploadFile to a NamedTemporaryFile and returns its path."""
    file_ext = get_file_extension(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def cleanup_tempfile(tmp_path: str) -> None:
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)
