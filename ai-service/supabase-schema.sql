create table if not exists public.mixing_worklogs (
  client_key text primary key,
  job_name text not null,
  material_type text,
  total_kg numeric not null default 0,
  temperature_c numeric,
  humidity_pct numeric,
  worker_name text,
  result text,
  overall_note text,
  cycles jsonb not null default '[]'::jsonb,
  cycle_notes jsonb not null default '[]'::jsonb,
  model_version text,
  created_at timestamptz not null default now()
);

alter table public.mixing_worklogs enable row level security;

drop policy if exists "field users can insert worklogs" on public.mixing_worklogs;
create policy "field users can insert worklogs"
  on public.mixing_worklogs for insert
  to authenticated
  with check (true);

drop policy if exists "field users can read worklogs" on public.mixing_worklogs;
create policy "field users can read worklogs"
  on public.mixing_worklogs for select
  to authenticated
  using (true);

drop policy if exists "field users can update worklogs" on public.mixing_worklogs;
create policy "field users can update worklogs"
  on public.mixing_worklogs for update
  to authenticated
  using (true)
  with check (true);
