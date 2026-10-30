"""
api/helpers/drive_helper.py
---------------------------
Reads documents out of a public Google Drive folder.

Uses the Drive v3 REST API directly over httpx, which is already a
dependency — the official google-api-python-client would add a large
dependency tree for two HTTP calls.

Auth model: a plain API key (GOOGLE_API_KEY). An API key can only read files
shared as "Anyone with the link". Private folders need a service account or
OAuth instead, which is a larger change — see ingest_from_drive_ctrl.
"""

import os
import re
import tempfile

import httpx

from agents.langgraph_agent.utils.utils import logger

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

# Drive reports Google-native formats with their own mime types; those cannot
# be downloaded with alt=media and are skipped rather than failing the import.
SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/plain": ".txt",
}

# Guard rails so one enormous folder cannot hang a request or fill the disk.
MAX_FILES = 50
MAX_FILE_BYTES = 50 * 1024 * 1024   # 50 MB, matching the upload UI's limit
HTTP_TIMEOUT = 60


class DriveError(Exception):
    """Raised for problems the caller should surface to the user verbatim."""


def get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise DriveError(
            "GOOGLE_API_KEY is not set. Add it to .env — see the Google Drive "
            "import setup notes."
        )
    return key


def extract_folder_id(url_or_id: str) -> str:
    """
    Pull the folder id out of whatever the user pasted.

    Accepts the id on its own, or any of the URL shapes Drive hands out:
      https://drive.google.com/drive/folders/<id>
      https://drive.google.com/drive/folders/<id>?usp=sharing
      https://drive.google.com/drive/u/0/folders/<id>
      https://drive.google.com/open?id=<id>
    """
    value = (url_or_id or "").strip()
    if not value:
        raise DriveError("No Google Drive folder link provided.")

    patterns = [
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    # A bare id: Drive ids are long-ish and have no spaces or slashes.
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", value):
        return value

    raise DriveError(
        "That does not look like a Google Drive folder link. Expected something "
        "like https://drive.google.com/drive/folders/<id>"
    )


def list_folder_files(folder_id: str) -> list[dict]:
    """
    List the supported documents directly inside a folder.

    Returns [{id, name, mimeType, size}]. Subfolders are not traversed — a
    recursive walk would make the request time unbounded, and the UI promises
    "the PDFs in this folder", not "in this folder tree".
    """
    api_key = get_api_key()

    mime_clause = " or ".join(
        f"mimeType='{mime}'" for mime in SUPPORTED_MIME_TYPES
    )
    query = f"'{folder_id}' in parents and trashed=false and ({mime_clause})"

    params = {
        "q": query,
        "key": api_key,
        "fields": "files(id,name,mimeType,size)",
        "pageSize": MAX_FILES,
        # Needed for files on shared drives; harmless otherwise.
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.get(DRIVE_FILES_URL, params=params)
    except Exception as e:
        raise DriveError(f"Could not reach Google Drive: {e}") from e

    if response.status_code == 403:
        raise DriveError(
            "Google Drive refused the request (403). Either the Drive API is not "
            "enabled for this API key's project, or the key is restricted to "
            "other APIs."
        )
    if response.status_code == 404:
        raise DriveError(
            "Folder not found. Check the link, and make sure the folder is "
            'shared as "Anyone with the link".'
        )
    if response.status_code != 200:
        raise DriveError(
            f"Google Drive returned {response.status_code}: {response.text[:200]}"
        )

    files = response.json().get("files", [])
    logger.info("Drive → folder %s contains %d supported file(s)", folder_id, len(files))
    return files


def download_file(file_id: str, mime_type: str) -> str:
    """
    Download one Drive file to a temp path and return it.

    The caller is responsible for deleting the temp file (cleanup_tempfile).
    """
    api_key = get_api_key()
    suffix = SUPPORTED_MIME_TYPES.get(mime_type, "")

    params = {"alt": "media", "key": api_key, "supportsAllDrives": "true"}

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            with client.stream(
                "GET", f"{DRIVE_FILES_URL}/{file_id}", params=params
            ) as response:
                if response.status_code != 200:
                    body = response.read()[:200].decode("utf-8", "ignore")
                    raise DriveError(
                        f"Download failed ({response.status_code}): {body}"
                    )

                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                written = 0
                try:
                    with os.fdopen(fd, "wb") as out:
                        for chunk in response.iter_bytes():
                            written += len(chunk)
                            # Stop mid-stream rather than after the fact, so an
                            # oversized file never lands on disk in full.
                            if written > MAX_FILE_BYTES:
                                raise DriveError(
                                    f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit"
                                )
                            out.write(chunk)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise

                return tmp_path

    except DriveError:
        raise
    except Exception as e:
        raise DriveError(f"Download failed: {e}") from e
