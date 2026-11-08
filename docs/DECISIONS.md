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
- A3: the API owns accounts (users table, argon2id hashes, POST /auth/login) and seeds the first admin; the web app never reads password hashes. ADMIN_EMAIL/ADMIN_PASSWORD are backend-only.
- A3: Auth.js v5's default session token is encrypted (JWE), which the API cannot verify; it is replaced with an HS256-signed JWT using the shared AUTH_SECRET, as the task list specifies.
- A3: POST /auth/login is not public either: the web server signs a 60-second "login" token with AUTH_SECRET to call it, so only /health is reachable without a token.
- A3: the browser calls the API through a same-origin Next.js proxy (/api/backend/*) that adds the session token from its httpOnly cookie; the browser never holds the token or calls the API origin.
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
