# Decisions

One line per decision not covered by the Technical Findings and Task List (7 Nov 2026).
Frontend-only decisions are in the frontend repository's docs/DECISIONS.md.

- Relational store: Supabase is already Postgres, so it is reused (no Neon). Existing tables keep the column name `organization_id` rather than `org_id`.
- A1: besides GROQ_API_KEY, the API also refuses to start without QDRANT_URL, SUPABASE_URL and SUPABASE_KEY; it cannot serve any page without them.
- A1: /health caches the provider check for 5 minutes, because Fly probes it every 30 s (about 288 provider calls a day instead of 2,900).
- A2: one extra error code, `invalid_request`, for malformed requests that are not file problems (for example an empty question); `invalid_file` is kept for uploads.
- A2: the 30 s hard timeout applies to the whole chat turn (router, grading, generation, checks), and each single provider attempt also times out at 30 s.
- A2: provider 401 (expired or invalid key) is reported as `provider_unavailable` and is not retried; only 429, 5xx, timeouts and connection errors are retried.
- OCR: Tesseract replaces the Fireworks vision reader in E1, as the task list decides.
- Google Drive import (/ingest/drive) is left untouched until the task list is complete.
- File storage for originals (E3): Supabase Storage.
- B2: chat_events has one extra outcome, `internal`, for unexpected server errors, so they are not miscounted as provider errors.
- B2: chat_events also stores the question text (needed to group B3's Unanswered questions) and the request_id.
- B2: following the metric definition literally, a reply with no citation (including greetings and small talk) is `no_context`.
- B2: `model` records every model used in the turn, comma-separated (router, grader and generator can differ); tokens are summed over all of them.
- Schema changes ship as numbered files in supabase/migrations/, applied in Supabase's SQL Editor.
- A3 (changed at the owner's request, 9 Nov 2026): no Auth.js and no shared AUTH_SECRET. The API checks email and password against the Supabase users table (argon2id) and starts a database session: a random 256-bit token whose SHA-256 hash is stored in auth_sessions (migration 003), valid 8 h. The web app keeps the token in an httpOnly, Secure, SameSite=Lax cookie and forwards it as a Bearer token through its same-origin proxy (/api/backend/*). Sign-out revokes the session; a password change revokes the user's other sessions.
- A3: the API owns accounts and seeds the first admin from ADMIN_EMAIL/ADMIN_PASSWORD (backend-only); the web app never reads password hashes.
- A3: verified sessions are cached in memory for 30 s, so a revoked session stops working within 30 s; expiry is always checked exactly. If the session store is unreachable, requests fail closed.
- A3: /docs, /redoc and /openapi.json are disabled unless ENABLE_API_DOCS=true (set locally only).
- A4: the Qdrant tenant field keeps its existing name, `organization_id`, instead of `org_id`; the JWT claim is `org_id`.
- A4: `org_filter` lives in data/qdrant_store.py, together with every Qdrant call (query, scroll, upsert, delete, indexes, ping), because the agent, ingestion and API layers all need it.
- A4: "migrate existing data into a default org" needed no data move: the HR_Policy chunks, chats and settings were already stored under `matz-demo-org`, which is now the seeded admin's organization.
- A4: data/knowledge_base/upload_to_qdrant.py (the old template-corpus seeder) is removed; it wrote to Qdrant directly and C5 replaces that corpus.
- A4: store.delete() refuses a call with only the organization condition, so no bug can wipe a whole organization's chunks.
- F1: the isolation suite runs the real controllers, store and agent nodes against in-memory fakes of Qdrant and Supabase (tests/fakes.py), so CI needs no services or secrets.
- F1: replacing or deleting a document now returns not_found unless the caller's organization has it, so a guessed id from another org is refused instead of being silently written into the caller's org.
- F1: the GitHub Actions workflow (.github/workflows/tests.yml) needs no secrets; it was also verified locally with no .env and an empty environment.
- B4: passwords must be 10 to 128 characters; changing one requires the current password. There is no complexity rule beyond length.
- A5: rate-limit windows are kept in memory in the API process (no Redis, per the task list); with more than one API machine each keeps its own counts. A refused request is not counted, so a blocked user recovers on time.
- A5: the per-IP limit uses the first X-Forwarded-For entry (set by the web app proxy from Vercel's client IP), else the socket peer. A direct caller can spoof that header, so the per-user limits are the firm ones; /health is exempt so uptime probes are never limited.
- A5: replacing a document counts as an upload, and a Drive import counts as one upload.
- A5: a PDF that opens with an empty password (owner restrictions only) is accepted; only a PDF that needs a password to open is refused. Encrypted Office files are detected by their OLE container.
- A5: a .txt upload must be UTF-8 without NUL bytes.
- A5: a replacement file is validated and extracted before the old chunks are deleted, so a rejected replacement keeps the current version.
