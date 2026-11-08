-- 001_chat_events.sql — task B2: one row per chat attempt, including failures.
--
-- Apply once: Supabase dashboard → SQL Editor → paste → Run. Safe to re-run.
-- Until it is applied, the API keeps working and logs
-- "record_chat_event failed" instead of writing rows.

create table if not exists public.chat_events (
    id                   uuid primary key default gen_random_uuid(),
    organization_id      text not null,
    user_id              text,              -- null until authentication (A3)
    session_id           uuid,              -- no FK: events outlive deleted chats
    outcome              text not null check (outcome in (
                             'answered', 'no_context', 'provider_error',
                             'timeout', 'rate_limited', 'internal')),
    latency_ms           integer not null,
    collections_searched text[] not null default '{}',
    cited_doc_ids        text[] not null default '{}',
    model                text,
    prompt_tokens        integer not null default 0,
    completion_tokens    integer not null default 0,
    question             text,              -- groups Unanswered questions (B3)
    request_id           text,
    created_at           timestamptz not null default now()
);

create index if not exists chat_events_org_created_idx
    on public.chat_events (organization_id, created_at desc);
create index if not exists chat_events_org_outcome_idx
    on public.chat_events (organization_id, outcome, created_at desc);

alter table public.chat_events enable row level security;
