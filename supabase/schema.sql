-- MATZ AI Knowledge Assistant — Supabase schema
--
-- Run once in a fresh project: Supabase dashboard → SQL Editor → paste → Run.
-- Derived from the queries in agents/langgraph_agent/utils/utils.py.
--
-- The backend should use the service_role key (SUPABASE_KEY), which bypasses
-- RLS. RLS is enabled with no policies so the public anon key can't read
-- anything.

create extension if not exists pgcrypto;

-- ── Sessions ─────────────────────────────────────────────────────────────────
create table if not exists public.sessions (
    id              uuid primary key default gen_random_uuid(),
    organization_id text not null,
    created_at      timestamptz not null default now()
);
create index if not exists sessions_org_created_idx
    on public.sessions (organization_id, created_at desc);

-- ── Messages ─────────────────────────────────────────────────────────────────
-- Deleting a session deletes its messages (delete_session relies on this).
create table if not exists public.messages (
    id               uuid primary key default gen_random_uuid(),
    session_id       uuid not null references public.sessions (id) on delete cascade,
    role             text not null check (role in ('user', 'assistant', 'system')),
    content          text not null,
    citations        jsonb not null default '[]'::jsonb,
    is_grounded      boolean,
    response_time_ms integer,
    created_at       timestamptz not null default now()
);
create index if not exists messages_session_created_idx
    on public.messages (session_id, created_at);
create index if not exists messages_created_idx
    on public.messages (created_at);

-- ── Collections ──────────────────────────────────────────────────────────────
create table if not exists public.collections (
    id              uuid primary key default gen_random_uuid(),
    organization_id text not null,
    name            text not null,
    description     text not null default '',
    icon            text not null default 'folder',
    created_at      timestamptz not null default now(),
    unique (organization_id, name)
);

-- ── Workspace settings (one row per org; upserted on organization_id) ───────
create table if not exists public.workspace_settings (
    organization_id text primary key,
    name            text not null default '',
    updated_at      timestamptz not null default now()
);

-- ── Assistant config (one row per org; upserted on organization_id) ─────────
create table if not exists public.assistant_config (
    organization_id text primary key,
    name            text not null default 'MATZ Assistant',
    personality     text not null default 'Helpful and precise',
    instructions    text not null default '',
    updated_at      timestamptz not null default now()
);

-- ── RPC: count user questions for an org (count_user_questions) ─────────────
create or replace function public.count_org_user_questions(org_id text)
returns bigint
language sql
stable
as $$
    select count(*)
    from public.messages m
    join public.sessions s on s.id = m.session_id
    where s.organization_id = org_id
      and m.role = 'user';
$$;

-- ── RPC: an org's messages since a timestamp (get_analytics_summary) ────────
create or replace function public.org_messages_since(org_id text, since timestamptz)
returns setof public.messages
language sql
stable
as $$
    select m.*
    from public.messages m
    join public.sessions s on s.id = m.session_id
    where s.organization_id = org_id
      and m.created_at >= since
    order by m.created_at;
$$;

-- ── Row level security ───────────────────────────────────────────────────────
alter table public.sessions           enable row level security;
alter table public.messages           enable row level security;
alter table public.collections        enable row level security;
alter table public.workspace_settings enable row level security;
alter table public.assistant_config   enable row level security;
