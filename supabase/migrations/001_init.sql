-- RinkIQ online: job queue + storage buckets.
--
-- Run once in the Supabase SQL editor (Dashboard → SQL → New query → paste → Run).
-- Safe to re-run. Access model: NOTHING is readable or writable with the public (anon) key. The website's
-- server routes and the Python worker use the service-role key, which bypasses row-level security.

-- ---------------------------------------------------------------- jobs
create table if not exists public.jobs (
  id            uuid primary key default gen_random_uuid(),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  status        text not null default 'uploading'
                check (status in ('uploading', 'queued', 'running', 'done', 'failed', 'canceled')),
  filename      text not null,
  video_path    text not null,                  -- object path inside the `videos` bucket
  size_bytes    bigint,
  options       jsonb not null default '{}'::jsonb,   -- start_s, seconds, quality, render, team_a, team_b, season …
  progress      real not null default 0 check (progress >= 0 and progress <= 1),
  message       text,                           -- what the worker is doing right now
  result        jsonb,                          -- stars, team table, tiles … (see worker/worker.py)
  error         text,
  worker        text,
  attempts      int not null default 0,
  started_at    timestamptz,
  finished_at   timestamptz,
  heartbeat_at  timestamptz
);

create index if not exists jobs_status_created_idx on public.jobs (status, created_at);
create index if not exists jobs_created_idx on public.jobs (created_at desc);

create or replace function public.touch_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end $$;

drop trigger if exists jobs_touch on public.jobs;
create trigger jobs_touch before update on public.jobs
  for each row execute function public.touch_updated_at();

-- Row-level security ON with no policies = deny everything to anon / authenticated.
alter table public.jobs enable row level security;
revoke all on table public.jobs from public, anon, authenticated;
grant all on table public.jobs to service_role;

-- ---------------------------------------------------------------- claiming work
-- Atomically hand the oldest waiting job to one worker. `for update skip locked` means two workers can
-- never take the same job. A job whose worker went silent for p_stale_seconds is handed out again
-- (up to 3 attempts), after which it is marked failed.
create or replace function public.claim_job(p_worker text, p_stale_seconds int default 300)
returns setof public.jobs
language plpgsql security definer set search_path = public as $$
declare
  j public.jobs;
begin
  update public.jobs
     set status = 'failed', error = 'The worker stopped repeatedly while processing this video.',
         finished_at = now()
   where status = 'running'
     and heartbeat_at < now() - make_interval(secs => p_stale_seconds)
     and attempts >= 3;

  select * into j from public.jobs
   where status = 'queued'
      or (status = 'running' and heartbeat_at < now() - make_interval(secs => p_stale_seconds))
   order by created_at
   limit 1
   for update skip locked;

  if not found then
    return;
  end if;

  update public.jobs
     set status = 'running',
         worker = p_worker,
         attempts = attempts + 1,
         started_at = coalesce(started_at, now()),
         heartbeat_at = now(),
         progress = case when status = 'queued' then 0 else progress end,
         message = 'Picked up by a worker',
         error = null
   where id = j.id
   returning * into j;

  return next j;
end $$;

revoke all on function public.claim_job(text, int) from public, anon, authenticated;
grant execute on function public.claim_job(text, int) to service_role;

-- ---------------------------------------------------------------- storage
-- Private buckets: files are only reachable through short-lived signed URLs made by the server.
-- 50 MB is the free-plan ceiling per file. Raise both this number and Storage → Settings → "Upload file
-- size limit" on a paid plan.
insert into storage.buckets (id, name, public, file_size_limit)
values ('videos', 'videos', false, 52428800),
       ('results', 'results', false, 524288000)
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit;
