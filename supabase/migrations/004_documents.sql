-- 004_documents.sql — task B1: one row per document, so "documents added"
-- per day comes from the database. E1 (statuses) and E7 (version, updated
-- by/at) build on this table.
--
-- Apply once: Supabase dashboard → SQL Editor → paste → Run. Safe to re-run.
-- Chunks stay in Qdrant; this table holds document-level facts only.
-- A deleted document keeps its row (deleted_at set) so past activity does
-- not change when a document is removed.

create table if not exists public.documents (
    organization_id text not null,
    id              text not null,             -- the document_id in Qdrant payloads
    title           text not null default '',
    collection_id   text,
    collection_name text,
    version         integer not null default 1,
    created_by      text,
    created_at      timestamptz not null default now(),
    updated_by      text,
    updated_at      timestamptz not null default now(),
    deleted_at      timestamptz,
    primary key (organization_id, id)
);

create index if not exists documents_org_created_idx
    on public.documents (organization_id, created_at desc);

alter table public.documents enable row level security;
