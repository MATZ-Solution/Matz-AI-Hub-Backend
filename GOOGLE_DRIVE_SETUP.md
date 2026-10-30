# Google Drive folder import — setup

Paste a Google Drive folder link on the Knowledge page and every supported
document in it is ingested into the knowledge base.

Supported: **PDF, DOCX, PPTX, XLSX, TXT**. Google-native formats (Google Docs,
Sheets, Slides) are skipped — they need a different export call.

---

## What you need to do

Everything below is in the Google Cloud console. No code changes.

### 1. Enable the Drive API

Google Cloud console → pick the **matz-ai-hub** project →
**APIs & Services → Library** → search "Google Drive API" → **Enable**.

### 2. Create an API key

**APIs & Services → Credentials → Create credentials → API key**.

Then click the new key and restrict it:

- **API restrictions** → *Restrict key* → tick **Google Drive API** only.
- Leave Application restrictions as *None* (the key is used server-side, so an
  HTTP-referrer restriction would block it).

> An **API key** is the right credential here — a plain string starting
> `AIza...`. The OAuth client ID/secret pair is a different thing: it is for
> "Sign in with Google" consent flows, and will not work for this.

### 3. Put it in the backend environment

Local — `Backend/.env`:

```
GOOGLE_API_KEY=AIza...
```

Render — **Environment → Add environment variable**, same name and value.
(`.env` is gitignored; never commit the key.)

### 4. Share the folder

In Drive: right-click the folder → **Share** → *General access* →
**Anyone with the link** → role **Viewer**.

An API key can only read publicly-shared files. This is why the folder must be
public — see *Limitations* below.

### 5. Use it

Knowledge page → **Import from Drive** → paste the folder link, choose a
collection, **Import folder**.

---

## How it works

`POST /ingest/drive` with `{folder_url, collection_name, collection_id}`.

1. The folder id is parsed out of the link (all the usual Drive URL shapes work)
2. Drive API v3 lists the supported files in that folder
3. Each file is downloaded to a temp file, then run through the **existing**
   ingestion pipeline — `extract_text` → `chunk_text` → `ingest_chunks`
4. A per-file result is returned, so a partial import still reports what worked

Only the Drive REST API is used, over `httpx`, so there is **no new Python
dependency**.

### Re-running is safe

Each file's `document_id` is `gdrive-<driveFileId>`, and existing points for
that id are deleted before re-ingesting. Importing the same folder twice
updates the documents rather than duplicating every chunk.

---

## Limitations

| | |
|---|---|
| **Public folders only** | An API key cannot read private files. Private folders need a service account (share the folder with its robot email) or OAuth. Both are a bigger change. |
| **No subfolders** | Only files directly in the folder. A recursive walk would make the request time unbounded. |
| **50 files max** | `MAX_FILES` in `api/helpers/drive_helper.py`. |
| **50 MB per file** | `MAX_FILE_BYTES`, matching the upload UI. Aborted mid-download, so an oversized file never lands on disk. |
| **Synchronous** | Fine for a handful of documents. A folder large enough to hit the request timeout would need a background job with a status endpoint. |

Because it runs synchronously, expect the request to take roughly as long as
uploading the same files by hand — and longer on Render's free tier, which
sleeps after 15 minutes idle and takes about a minute to wake.

---

## Troubleshooting

Messages are written to be shown to the user as-is.

| Message | Cause |
|---|---|
| `GOOGLE_API_KEY is not set` | Step 3 — missing from `.env` / Render env, or the service was not restarted |
| `Google Drive refused the request (403)` | Drive API not enabled (step 1), or the key is restricted to other APIs (step 2) |
| `Folder not found` | Wrong link, or the folder is not shared as "Anyone with the link" (step 4) |
| `That does not look like a Google Drive folder link` | A file link rather than a folder link — the URL needs `/folders/` |
| Import succeeds but `files_found: 0` | The folder has no supported file types, or only Google-native Docs/Sheets/Slides |

---

## Files in this change

**Backend**

- `api/helpers/drive_helper.py` — **new**; link parsing, folder listing, download
- `api/controllers/knowledge_controller.py` — adds `ingest_from_drive_ctrl`
- `api/routes/knowledge_routes.py` — adds `POST /ingest/drive`
- `api/schemas/schemas.py` — adds `DriveIngestRequest` / `DriveFileResult` / `DriveIngestResponse`

**Frontend**

- `lib/agent-api.ts` — adds `ingestFromDrive` and its types
- `lib/queries.ts` — adds `useIngestFromDrive` (invalidates documents, stats, usage, collections)
- `app/admin/knowledge/page.tsx` — adds the "Import from Drive" button and modal

---

## Status

The link parsing, folder listing, error handling and download limits were
tested against a mocked Drive API, and both frontend and backend compile
clean. **It has not been run against the real Drive API** — that needs the key
and a shared folder, so the first live run is the real test.
