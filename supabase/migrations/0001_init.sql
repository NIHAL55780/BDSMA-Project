-- Aggregate experiment results only. Never store raw or per-user check-ins.

create table datasets (
  id text primary key,
  name text not null,
  n_users int,
  n_pois int,
  n_checkins int
);

create table mechanisms (
  id text primary key,
  name text not null,
  acts_at text not null,
  param_name text
);

create table runs (
  id uuid primary key default gen_random_uuid(),
  dataset_id text references datasets(id),
  mechanism_id text references mechanisms(id),
  task text not null,
  param_value double precision,
  epsilon_reported double precision,
  seed int not null,
  created_at timestamptz default now()
);

create table metrics (
  run_id uuid references runs(id) on delete cascade,
  name text not null,
  value double precision not null,
  primary key (run_id, name)
);

create table compliance_items (
  id text primary key,
  obligation text not null,
  act_reference text not null,
  control text not null,
  test text not null
);

create table compliance_status (
  item_id text references compliance_items(id),
  mechanism_id text references mechanisms(id),
  dataset_id text references datasets(id),
  status text not null check (status in ('pass','partial','fail','na')),
  evidence text,
  primary key (item_id, mechanism_id, dataset_id)
);

create table audit_log (
  id bigserial primary key,
  at timestamptz default now(),
  method text not null,
  path text not null,
  detail text
);

alter table datasets enable row level security;
alter table mechanisms enable row level security;
alter table runs enable row level security;
alter table metrics enable row level security;
alter table compliance_items enable row level security;
alter table compliance_status enable row level security;
alter table audit_log enable row level security;

create policy "public read" on datasets for select using (true);
create policy "public read" on mechanisms for select using (true);
create policy "public read" on runs for select using (true);
create policy "public read" on metrics for select using (true);
create policy "public read" on compliance_items for select using (true);
create policy "public read" on compliance_status for select using (true);
-- audit_log is service-role only (no public policy)
