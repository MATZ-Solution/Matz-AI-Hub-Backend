-- 003_auth_sessions.sql — database-backed sign-in sessions (replaces the
-- shared-secret JWT from A3, so no AUTH_SECRET is needed anywhere).
--
-- Apply once: Supabase dashboard → SQL Editor → paste → Run. Safe to re-run.
--
-- The browser holds a random 256-bit token in an httpOnly cookie; only its
-- SHA-256 hash is stored here, so a leaked table cannot be used to sign in.

create table if not exists public.auth_sessions (
    id              uuid primary key default gen_random_uuid(),
    token_hash      text not null unique,
    user_id         uuid not null references public.users (id) on delete cascade,
    organization_id text not null,
    expires_at      timestamptz not null,
    revoked_at      timestamptz,
    created_at      timestamptz not null default now(),
    last_seen_at    timestamptz
);

create index if not exists auth_sessions_user_idx on public.auth_sessions (user_id);
create index if not exists auth_sessions_expires_idx on public.auth_sessions (expires_at);

alter table public.auth_sessions enable row level security;
