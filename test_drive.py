"""
test_drive.py — exercise the Google Drive import pipeline from the terminal.

Run from Backend/:

    python test_drive.py                      # parsing + config checks only
    python test_drive.py <folder-link>        # also list the folder
    python test_drive.py <folder-link> --full # also download + extract page 1

Stops at the first failure and says what to fix. Nothing is written to Qdrant,
so this is safe to run against a real folder.
"""

import sys
import os

from dotenv import load_dotenv

load_dotenv()

from api.helpers.drive_helper import (          # noqa: E402
    DriveError, extract_folder_id, list_folder_files, download_file,
    SUPPORTED_MIME_TYPES, MAX_FILES, MAX_FILE_BYTES,
)
OK, BAD, INFO = "[ OK ]", "[FAIL]", "[ .. ]"


def cleanup_tempfile(path):
    """Local copy — importing api.helpers.file_helper would drag in FastAPI
    for what is one os.remove."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def step(n, title):
    print(f"\n{'=' * 60}\n{n}. {title}\n{'=' * 60}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    folder_link = args[0] if args else None

    # ── 1. Link parsing (no network, no key) ─────────────────────────────
    step(1, "Link parsing")
    samples = [
        "https://drive.google.com/drive/folders/1AbC_dEfGhIjKlMnOpQ",
        "https://drive.google.com/drive/folders/1AbC_dEfGhIjKlMnOpQ?usp=sharing",
        "https://drive.google.com/drive/u/0/folders/1AbC_dEfGhIjKlMnOpQ",
        "https://drive.google.com/open?id=1AbC_dEfGhIjKlMnOpQ",
    ]
    for s in samples:
        try:
            print(f"  {OK} {extract_folder_id(s)}  <-  {s[:52]}")
        except DriveError as e:
            print(f"  {BAD} {s}\n        {e}")
            return 1

    try:
        extract_folder_id("not a link")
        print(f"  {BAD} junk input was accepted — it should raise")
        return 1
    except DriveError:
        print(f"  {OK} junk input correctly rejected")

    # ── 2. Configuration ─────────────────────────────────────────────────
    step(2, "Configuration")
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print(f"  {BAD} GOOGLE_API_KEY is not set")
        print("        Add it to Backend/.env — see GOOGLE_DRIVE_SETUP.md")
        return 1
    print(f"  {OK} GOOGLE_API_KEY loaded ({key[:6]}..., {len(key)} chars)")
    if not key.startswith("AIza"):
        print("        NOTE: Drive API keys normally start with 'AIza'.")
        print("        An OAuth client id/secret will NOT work here.")
    print(f"  {INFO} limits: {MAX_FILES} files, {MAX_FILE_BYTES // (1024 * 1024)} MB each")
    print(f"  {INFO} types : {', '.join(sorted(SUPPORTED_MIME_TYPES.values()))}")

    if not folder_link:
        print("\nPass a folder link to test listing:")
        print("  python test_drive.py https://drive.google.com/drive/folders/...")
        return 0

    # ── 3. List the folder ───────────────────────────────────────────────
    step(3, "Listing the folder")
    try:
        folder_id = extract_folder_id(folder_link)
        print(f"  {INFO} folder id: {folder_id}")
        files = list_folder_files(folder_id)
    except DriveError as e:
        print(f"  {BAD} {e}")
        return 1

    if not files:
        print(f"  {BAD} no supported files found")
        print("        Either the folder is empty, or it only holds Google-native")
        print("        Docs/Sheets/Slides, which are not supported.")
        return 1

    print(f"  {OK} {len(files)} file(s):")
    for f in files:
        size = f.get("size")
        size_txt = f"{int(size) / 1024:.0f} KB" if size else "size unknown"
        print(f"        - {f['name']}  ({size_txt})")

    if not full:
        print("\nAdd --full to download the first file and extract its text.")
        return 0

    # ── 4. Download + extract (the actual pipeline) ──────────────────────
    step(4, "Download + text extraction")
    target = files[0]
    print(f"  {INFO} using: {target['name']}")

    tmp_path = None
    try:
        tmp_path = download_file(target["id"], target.get("mimeType", ""))
        print(f"  {OK} downloaded ({os.path.getsize(tmp_path)} bytes)")

        from data.pipeline.processor import extract_text
        from data.pipeline.chunker import chunk_text

        result = extract_text(tmp_path)
        if not result["success"]:
            print(f"  {BAD} extraction failed: {result['error']}")
            return 1

        print(f"  {OK} extracted {len(result['text'])} chars "
              f"from {result['page_count']} page(s) via {result['method']}")
        if result.get("scanned_pages"):
            print(f"  {INFO} pages with no text layer: {result['scanned_pages']}")

        chunks = chunk_text(
            text=result["text"],
            document_id=f"gdrive-{target['id']}",
            page_count=result["page_count"],
        )
        print(f"  {OK} chunked into {len(chunks)} piece(s)")
        print(f"\n  --- first 300 chars ---\n{result['text'][:300]}\n  ---")

    except DriveError as e:
        print(f"  {BAD} {e}")
        return 1
    finally:
        cleanup_tempfile(tmp_path)

    print("\nPipeline works. Nothing was written to Qdrant — use the UI or the")
    print("/ingest/drive endpoint for a real import.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
