-- 002_users.sql — task A3: accounts for email + password sign-in.
--
-- Apply once: Supabase dashboard → SQL Editor → paste → Run. Safe to re-run.
-- The API creates the first admin from ADMIN_EMAIL / ADMIN_PASSWORD on
-- startup once this table exists.

create table if not exists public.users (
    id              uuid primary key default gen_random_uuid(),
    organization_id text not null,
    email           text not null,
    name            text not null default '',
    password_hash   text not null,             -- argon2id, never plain text
    role            text not null default 'member'
                        check (role in ('admin', 'manager', 'member')),
    is_active       boolean not null default true,
    failed_logins   integer not null default 0, -- lockout (A8)
    locked_until    timestamptz,                -- lockout (A8)
    last_login_at   timestamptz,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

-- One account per email, compared case-insensitively.
create unique index if not exists users_email_lower_idx on public.users (lower(email));
create index if not exists users_org_idx on public.users (organization_id);

alter table public.users enable row level security;
