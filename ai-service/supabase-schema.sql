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

-- 운영 시스템의 기본 작업 단위입니다. 아직 AI 학습을 하지 않아도
-- 대시보드·작업일지·배합 이력이 같은 작업 ID를 사용할 수 있도록 합니다.
create table if not exists public.work_orders (
  id uuid primary key default gen_random_uuid(),
  client_key text unique not null,
  job_name text not null,
  material_type text,
  total_kg numeric not null default 0,
  temperature_c numeric,
  humidity_pct numeric,
  worker_name text,
  status text not null default '작업 준비',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.mixing_cycles (
  id uuid primary key default gen_random_uuid(),
  work_order_id uuid not null references public.work_orders(id) on delete cascade,
  cycle_no integer not null,
  round_no integer not null,
  amount_kg numeric not null default 0,
  water_l numeric,
  mixing_min numeric,
  hopper_wait_min numeric,
  retarder_used boolean not null default false,
  risk_pct numeric,
  actual_water_l numeric,
  actual_mixing_min numeric,
  note text,
  created_at timestamptz not null default now(),
  unique(work_order_id, cycle_no, round_no)
);

create table if not exists public.image_inspections (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid(),
  work_order_id uuid references public.work_orders(id) on delete set null,
  image_path text not null,
  job_name text,
  material_type text,
  cycle_no integer,
  round_no integer,
  batch_kg numeric,
  temperature_c numeric,
  humidity_pct numeric,
  hopper_wait_min numeric,
  retarder_used boolean,
  analysis_status text not null default 'AI 분석 대기',
  hardening_score numeric,
  crack_score numeric,
  analysis_result text,
  improvement_note text,
  model_version text,
  created_at timestamptz not null default now()
);

alter table public.mixing_worklogs enable row level security;
alter table public.work_orders enable row level security;
alter table public.mixing_cycles enable row level security;
alter table public.image_inspections enable row level security;

drop policy if exists "Users insert own inspections" on public.image_inspections;
drop policy if exists "Users view own inspections" on public.image_inspections;
drop policy if exists "Users update own inspections" on public.image_inspections;

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

drop policy if exists "field users can manage work orders" on public.work_orders;
create policy "field users can manage work orders"
  on public.work_orders for all to authenticated
  using (true) with check (true);

drop policy if exists "field users can manage mixing cycles" on public.mixing_cycles;
create policy "field users can manage mixing cycles"
  on public.mixing_cycles for all to authenticated
  using (true) with check (true);

drop policy if exists "field users can manage image inspections" on public.image_inspections;
create policy "field users can manage image inspections"
  on public.image_inspections for all to authenticated
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

insert into storage.buckets (id, name, public)
values ('refractory-images', 'refractory-images', false)
on conflict (id) do nothing;

drop policy if exists "field users can upload refractory images" on storage.objects;
drop policy if exists "Users upload own refractory images" on storage.objects;
drop policy if exists "Users view own refractory images" on storage.objects;
drop policy if exists "Users update own refractory images" on storage.objects;
create policy "field users can upload refractory images"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'refractory-images' and (storage.foldername(name))[1] = auth.uid()::text);

drop policy if exists "field users can read refractory images" on storage.objects;
create policy "field users can read refractory images"
  on storage.objects for select to authenticated
  using (bucket_id = 'refractory-images' and (storage.foldername(name))[1] = auth.uid()::text);
