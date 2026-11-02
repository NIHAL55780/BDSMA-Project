# Project Implementation Plan
## Privacy-Utility Trade-off in Social Location Analytics under India's DPDP Act

Course: Big Data and Social Media Analytics (Sem 7), Module 6, Paper 6
Companion to: `Literature_Survey_Location_Privacy_DPDP.docx` (paper numbers like [12] below refer to that survey)

---

## 0. Read this first (for the implementer and any coding agent)

- **What we are building:** an experiment pipeline that runs six privacy mechanisms on the same location datasets and the same two tasks, attacks every output, and reports utility against measured privacy risk. A small web dashboard then displays the results and a DPDP compliance checklist.
- **The research contribution is the experiment pipeline** (Python). The FastAPI + React + Supabase app is the presentation layer. Build the pipeline first; the app only reads its results.
- **Work phase by phase (Section 10).** Each phase has acceptance criteria. Do not start the next phase until the current one passes.
- **Never invent numbers.** Every number in a figure, table or the paper must come from a run logged in `results/`. The abstract has placeholders (`[utility retained at moderate epsilon]`, `[mechanism]`) that are filled only after Phase 7.
- **Determinism:** every script takes `--seed`. Use seeds `0, 1, 2` and report mean ± std.
- **Config, not code edits:** all hyperparameters live in `experiments/configs/*.yaml`.

---

## 1. Problem in one paragraph

Location check-in and trajectory data supports mobility, urban and public-safety analytics but is highly re-identifiable [11]. Privacy techniques (cloaking, k-anonymity, geo-indistinguishability, DP-SGD, federated learning) are usually evaluated one at a time, on different tasks and datasets, and often report utility only. India's DPDP Act, 2023 sets obligations (consent, purpose limitation, data minimisation, security safeguards, erasure) but does not name technical controls. We compare the mechanisms on shared tasks and data, measure privacy with real attacks, and map the results to DPDP obligations.

---

## 2. Gaps → what we build → evidence we produce

| # | Gap (from topic brief + survey) | What we build | Evidence / output |
|---|---|---|---|
| G1 | Mechanisms compared in isolation, on different tasks and data [1]-[6] | One harness: same datasets, same splits, same two tasks, six mechanisms + baselines | Unified results table; Fig. 5 (utility vs privacy parameter) |
| G2 | DPDP obligations not turned into technical criteria [14], [15] | `compliance/checklist.yaml`: obligation → section → control → test → pass rule | Fig. 9 matrix; checklist page in dashboard |
| G3 | Utility reported without attack-based privacy evaluation [7]-[9] | Attack suite: re-identification on released data, membership inference on trained models | Fig. 6 (attack success vs budget); Fig. 7 Pareto front |
| G4 | FL with vs without DP not compared for POI tasks [7]-[9], [12] | FedAvg and DP-FedAvg next to centralised and DP-SGD training | Federated vs centralised comparison table |

**Key method decision (important):** epsilon is not comparable across mechanisms. Geo-ind epsilon is per km per report, DP-SGD epsilon is per training example, DP-FedAvg epsilon is per user [3]. So the **common privacy axis on Pareto plots is empirical attack success**, and each mechanism's own parameter (epsilon, k, cell size) is shown only on its own curve.

---

## 3. Scope

### MVP (must have for submission)
- Datasets: **Foursquare NYC** and **Gowalla** (city-level subset).
- Tasks: next-POI prediction, hotspot detection.
- Mechanisms: M0–M7 in Section 5 (M7 only for hotspots).
- Attacks: A1 re-identification, A2 loss-threshold membership inference.
- Outputs: results CSVs, Figures 1–9, DPDP checklist, dashboard with results viewer + map demo + checklist.

### Stretch (only if MVP is done)
- More datasets: Foursquare Tokyo, Brightkite, Porto taxi, Geolife, T-Drive (listed in the topic brief).
- Shadow-model membership inference; aggregate membership inference on hotspot counts [13].
- GETNext [10] as a second next-POI model.
- User-level DP-SGD; Gi* hotspot statistics; Flower-based FL.

---

## 4. Data

| Dataset | Type | Use | Source |
|---|---|---|---|
| Foursquare NYC (MVP) | Check-ins with venue lat/lon | Next-POI, hotspots | Dingqi Yang's Foursquare dataset page: https://sites.google.com/site/yangdingqi/home/foursquare-dataset |
| Gowalla (MVP, subset) | Check-ins | Next-POI, hotspots | SNAP: https://snap.stanford.edu/data/loc-gowalla.html |
| Brightkite (stretch) | Check-ins | Next-POI | SNAP: https://snap.stanford.edu/data/loc-brightkite.html |
| Porto taxi, Geolife, T-Drive (stretch) | GPS trajectories | Hotspots, re-identification | Official dataset pages (check licence before use) |

These are non-commercial research datasets. Keep raw data in `data/raw/` (gitignored). Never upload raw or per-user data to Supabase.

### Preprocessing (`lpdp/data/preprocess.py`)
1. Gowalla: keep check-ins inside one city bounding box (default: a large metro area; set in config) so it is comparable to NYC and fits on CPU/Colab.
2. Drop users with < 10 check-ins and POIs with < 10 visits; repeat until stable.
3. Re-index users and POIs to contiguous integers. Keep a POI table `poi_id, lat, lon, category`.
4. Project coordinates to metres (local UTM via `pyproj`) for all distance work.
5. Sort each user's check-ins by time; cut into sessions with a 24-hour gap rule.
6. **User split (for membership inference):** users → set A (70%, "members") and set B (30%, "non-members"), stratified by activity.
7. **Time split (for utility):** for set A, first 80% of each user's check-ins → train, next 10% → val, last 10% → test.
8. Save to `data/processed/{dataset}/` as Parquet: `checkins.parquet`, `pois.parquet`, `split.json`.

Acceptance: a `dataset_stats.json` with user / POI / check-in counts before and after filtering; these counts go into the paper's dataset table.

---

## 5. Privacy mechanisms (`lpdp/mechanisms/`)

| ID | Mechanism | Where it acts | Parameter sweep | Notes |
|---|---|---|---|---|
| M0 | No privacy (raw, centralised) | – | – | Upper bound on utility, baseline |
| M1 | Spatial cloaking (grid generalisation) | Client / data | cell size ∈ {250, 500, 1000, 2000} m | Replace each check-in by its cell; map cell to the POI nearest the cell centroid (representative POI) |
| M2 | k-anonymity cloaking | Data (trusted curator) | k ∈ {2, 5, 10, 20} | Smallest quadtree cell containing check-ins of ≥ k distinct users; then representative POI as in M1. This is also the "k-anonymity only" baseline in the brief |
| M3 | Geo-indistinguishability (Planar Laplace) [19] | Client / data | ε ∈ {0.5, 1, 2, 5, 10} per km | Perturb lat/lon, then snap to nearest POI in the public POI table |
| M4 | DP-SGD (centralised) [20] | Model training | ε ∈ {1, 2, 4, 8} (δ = 1/N_train_examples) | Opacus `PrivacyEngine.make_private_with_epsilon` |
| M5 | Federated learning, FedAvg [21] | Training (no raw data at server) | – | Each user in set A is one client |
| M6 | FL + DP (DP-FedAvg, client-level) | Training | noise multiplier z ∈ {0.5, 1.0, 1.5, 2.0}; report resulting ε | Clip each client update to norm C, add Gaussian noise N(0, z²C²) at server |
| M7 | Central DP histogram (hotspots only) [6] | Aggregate release | ε ∈ {0.1, 0.5, 1, 2, 5} | Laplace noise on grid counts; bound each user to ≤ c check-ins per release so sensitivity = c |

### Implementation notes
- **Planar Laplace sampling (M3):** θ ~ U[0, 2π); p ~ U[0, 1); r = −(1/ε) · (W₋₁((p − 1)/e) + 1), with `scipy.special.lambertw(x, k=-1).real`. ε in 1/metre internally (convert from per-km). Unit test: mean displacement ≈ 2/ε within 2% over 100k samples.
- **Next-POI inputs under M1–M3:** the user's history (model input) is privatised; the prediction target stays the true next POI. This models "privatise on device, still serve the real user".
- **DP-SGD (M4):** use `opacus.layers.DPGRU` (or `DPLSTM`) in place of `nn.GRU`. Opacus gives example-level DP. A user contributes many windows, so also cap windows per user at m (config) and report the group-privacy bound for user level. State this clearly in the paper; the attacks are user-level either way.
- **FedAvg / DP-FedAvg (M5, M6):** implement in plain PyTorch (`lpdp/fl/`). It is ~150 lines, deterministic and fast to simulate with ~1k clients. Per round: sample fraction q of clients (default 50 clients), 1 local epoch, aggregate. For M6 compute ε with Opacus `RDPAccountant`: call `accountant.step(noise_multiplier=z, sample_rate=q)` once per round, then `get_epsilon(delta=1/num_clients)`. Flower (`flwr`) is an optional stretch; if used, check the installed version's DP strategy wrappers.
- **Communication cost (FL):** log bytes per round = parameters × 4 bytes × clients per round (needed for the "communication cost" metric in the brief).

---

## 6. Utility tasks (`lpdp/tasks/`, `lpdp/models/`)

### T1 Next-POI prediction
- Model (MVP): GRU with POI embedding (dim 128) + hour-of-week embedding (dim 16) → hidden 256 → softmax over POIs. Input: last L = 20 check-ins in the session window.
- Train: Adam, lr 1e-3, early stopping on val Acc@10 (non-private runs). For DP runs, fix the number of epochs in config so ε is known in advance.
- Metrics: **Acc@1, Acc@5, Acc@10, MRR** on set A test windows.

### T2 Hotspot detection
- Grid: 500 m cells over the city bounding box. Time slice: whole period (MVP), weekday vs weekend (stretch).
- Hotspots = top 5% of cells by check-in count (MVP); Getis-Ord Gi* significant cells via `esda` (stretch).
- Compare the hotspot set from privatised data to the set from raw data: **hotspot F1** (primary), Spearman correlation of counts, mean absolute count error.
- Mechanisms that apply: M0, M1, M2, M3, M7. M4–M6 are model-training mechanisms and do not apply to T2; say so in the paper rather than forcing it.

---

## 7. Attack suite (`lpdp/attacks/`)

### A1 Re-identification (applies to released data: M0–M3)
- Threat model: attacker holds p raw points (lat/lon + timestamp) of a target, e.g. from their public social posts, plus the full released dataset with pseudonymous IDs.
- Procedure: for each of 500 sampled target users, score every released user by the number of attacker points that have a released point within d = 250 m and Δt = 1 h; predict the top-scoring user.
- Sweep p ∈ {2, 4, 8}. Metric: **top-1 re-identification rate** (also report top-5).
- Also report raw-data unicity (fraction of users uniquely identified by p points), as in [11].

### A2 Membership inference (applies to trained models: M0, M1–M3-trained models, M4, M5, M6)
- Members: set A users. Non-members: set B users (never trained on). Balanced sampling.
- Score per user = mean cross-entropy loss of the model on that user's windows (lower loss → more likely member). Loss-threshold attack.
- Metrics: **AUC** and **TPR at 1% FPR**.
- Stretch: shadow-model attack [22] trained on set B splits; trajectory-level attack as in [12].

### Risk thresholds (project policy, configurable, not legal standards)
- "Acceptable" release: A1 top-1 rate ≤ 5% at p = 4, and A2 AUC ≤ 0.55.
- These thresholds feed the checklist pass/fail for the "security safeguards" row. Put them in `configs/policy.yaml`.

---

## 8. DPDP compliance checklist (`lpdp/compliance/checklist.yaml`)

Indicative mapping for an academic project, not legal advice. Check section numbers against the official Gazette text and have faculty review it (the brief calls for "legal expert review"). Note: the DPDP Rules, 2025 were notified on 13 Nov 2025 and most data-fiduciary obligations apply from 13 May 2027.

| Obligation | DPDP Act ref. | Technical control in our system | How we test it (pass rule) |
|---|---|---|---|
| Lawful processing / consent | s.4, s.6 | Consent table `user_id, purpose, granted_at, withdrawn_at`; pipeline filters to users consenting for the run's purpose | Unit test: users without consent never appear in any derived file |
| Notice of purpose and data items | s.5 | Notice text generated from run config (purposes, fields, retention) | Notice file exists for every run config |
| Purpose limitation | s.5, s.6 (consent for a specified purpose) | Every run carries a `purpose` tag; loader refuses data for a purpose without consent | Unit test with a mismatched purpose fails loudly |
| Data minimisation | s.6(1) (data limited to what is necessary) | Keep only pseudonymous user ID, POI/coordinates, timestamp; drop all else; coordinate precision set per mechanism | Schema check: no extra columns; report precision used |
| Withdrawal and erasure | s.6(4), s.8(7), s.12 | `erase_user(user_id)` removes the user from raw, processed and released data; models retrained (machine unlearning noted as limitation) | Test: after erasure, user absent from every artefact |
| Reasonable security safeguards | s.8(5) | Pseudonymisation; mechanisms M1–M7; Supabase row-level security; no raw data in the cloud | Attack results meet `policy.yaml` thresholds |
| Breach intimation | s.8(6) | Out of scope for experiments; audit log of data access in the dashboard backend | Log entries exist for every data-touching endpoint |
| Children's data; no tracking or behavioural monitoring of children | s.9 (incl. s.9(3)) | Exclude any user flagged as a child; datasets have no age field, so document as a limitation | Filter exists and is tested on synthetic flags |
| Significant data fiduciary: DPIA and audit | s.10 | The evaluation report (utility + attack results per mechanism) serves as DPIA input | Report generated per dataset |
| Research / statistical exemption | s.17(2)(b) | Our experiments use public research datasets and make no decisions about specific individuals | Stated in paper; no per-user outputs published |

Output: Fig. 9, a matrix of obligations (rows) × mechanisms (columns) with cells = pass / partial / not applicable, computed from the checklist rules and results.

---

## 9. System architecture

```
                ┌──────────────────────── experiments/ (Python) ───────────────────────┐
 raw datasets → │ preprocess → mechanisms (M0–M7) → tasks (T1, T2) → attacks (A1, A2) │ → results/*.csv, figures/*.png
                └───────────────────────────────────────┬──────────────────────────────┘
                                                        │ scripts/sync_supabase.py (aggregates only)
                                                        ▼
                                   Supabase Postgres (runs, metrics, checklist) + Storage (figures)
                                                        │
                                          backend/ FastAPI (read API + live perturbation demo)
                                                        │
                                          frontend/ React dashboard (charts, map, checklist)
```

### Tech stack
| Layer | Choice | Why |
|---|---|---|
| Experiments | Python 3.11, PyTorch, Opacus, NumPy, pandas, SciPy, scikit-learn, pyproj, matplotlib, Typer, PyYAML, pytest | Standard for DP and FL research; Opacus is the PyTorch DP-SGD library |
| FL | Plain PyTorch simulation (MVP); Flower optional | Simple, deterministic, enough for simulated federation |
| Backend | FastAPI, Pydantic v2, uvicorn, `supabase` Python client | Serves results; can import `lpdp.mechanisms` for the live map demo |
| Database | Supabase (Postgres + Storage) | Stores run metadata, metrics, checklist, figure files. Holds no personal data |
| Frontend | React + Vite + TypeScript, Recharts, react-leaflet, TanStack Query, Tailwind | Charts for Pareto/curves, Leaflet for raw vs perturbed map |
| Compute | Laptop CPU or Google Colab GPU | Models are small |

How relevant is the web stack? FastAPI + React + Supabase is not needed for the research itself, but it is a good fit for the "design guidelines and open code" contribution: a dashboard where anyone can pick a use case and see the recommended mechanism, its trade-off and its DPDP status.

### Repository layout
```
location-privacy-dpdp/
├── experiments/
│   ├── lpdp/
│   │   ├── data/          download.py, preprocess.py, splits.py
│   │   ├── mechanisms/    cloaking.py, kanon.py, geoind.py, central_dp.py
│   │   ├── models/        gru_next_poi.py
│   │   ├── train/         centralised.py, dpsgd.py
│   │   ├── fl/            fedavg.py, dp_fedavg.py, accounting.py
│   │   ├── tasks/         next_poi.py, hotspot.py
│   │   ├── attacks/       reid.py, mia.py
│   │   ├── eval/          metrics.py, pareto.py
│   │   └── compliance/    checklist.yaml, evaluate.py
│   ├── configs/           base.yaml, foursquare_nyc.yaml, gowalla.yaml, sweep.yaml, policy.yaml
│   ├── scripts/           run_sweep.py, make_figures.py, sync_supabase.py
│   └── tests/
├── backend/               app/main.py, app/routers/, app/schemas.py
├── frontend/              src/pages/, src/components/
├── supabase/migrations/   0001_init.sql
├── results/               (gitignored except summary CSVs)
├── figures/
└── README.md
```

### Results format (`results/runs.csv`, one row per run)
`run_id, dataset, task, mechanism, param_name, param_value, epsilon_reported, seed, acc1, acc5, acc10, mrr, hotspot_f1, reid_top1_p2, reid_top1_p4, reid_top1_p8, mia_auc, mia_tpr_at_1fpr, comm_bytes_per_round, train_seconds, created_at`
Empty cells where a metric does not apply.

### Supabase schema (`supabase/migrations/0001_init.sql`)
```sql
create table datasets (
  id text primary key,            -- 'foursquare_nyc', 'gowalla_city'
  name text not null,
  n_users int, n_pois int, n_checkins int
);

create table mechanisms (
  id text primary key,            -- 'M0'..'M7'
  name text not null,
  acts_at text not null,          -- 'client' | 'curator' | 'model' | 'aggregate'
  param_name text
);

create table runs (
  id uuid primary key default gen_random_uuid(),
  dataset_id text references datasets(id),
  mechanism_id text references mechanisms(id),
  task text not null,             -- 'next_poi' | 'hotspot'
  param_value double precision,
  epsilon_reported double precision,
  seed int not null,
  created_at timestamptz default now()
);

create table metrics (
  run_id uuid references runs(id) on delete cascade,
  name text not null,             -- 'acc10', 'hotspot_f1', 'reid_top1_p4', 'mia_auc', ...
  value double precision not null,
  primary key (run_id, name)
);

create table compliance_items (
  id text primary key,            -- 'consent', 'minimisation', ...
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

-- Read-only public access for the dashboard
alter table datasets enable row level security;
alter table mechanisms enable row level security;
alter table runs enable row level security;
alter table metrics enable row level security;
alter table compliance_items enable row level security;
alter table compliance_status enable row level security;
create policy "public read" on datasets for select using (true);
create policy "public read" on mechanisms for select using (true);
create policy "public read" on runs for select using (true);
create policy "public read" on metrics for select using (true);
create policy "public read" on compliance_items for select using (true);
create policy "public read" on compliance_status for select using (true);
```
Writes happen only from `sync_supabase.py` using the service key kept in a local `.env` (never committed, never in the frontend).

### Backend API (FastAPI)
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status |
| GET | `/datasets`, `/mechanisms` | lists |
| GET | `/runs?dataset=&task=&mechanism=` | runs with metrics, averaged over seeds (mean, std) |
| GET | `/pareto?dataset=&task=&utility=acc10&risk=mia_auc` | all points + non-dominated set |
| GET | `/compliance?dataset=` | checklist items and status matrix |
| POST | `/perturb` | body `{mechanism, param, n_points}`; returns raw and perturbed points from a fixed **public sample** of POI locations (not user check-ins) for the map demo |
| POST | `/recommend` | body `{task, max_risk, min_utility}`; returns the mechanisms meeting both limits, ranked (rules built from results, Phase 8) |

### Frontend pages
1. **Overview:** problem, pipeline diagram (Fig. 1), datasets.
2. **Trade-off explorer:** dataset/task selectors; utility vs parameter curves (Fig. 5), attack success vs budget (Fig. 6), Pareto front (Fig. 7). Recharts.
3. **Map demo:** pick mechanism + parameter, see raw vs perturbed points on a Leaflet map (Fig. 8).
4. **DPDP checklist:** obligation × mechanism matrix with pass/partial/fail/na and evidence on hover (Fig. 9).
5. **Recommender:** use-case form → recommended mechanism with its numbers.

---

## 10. Phased plan with acceptance criteria

| Phase | Work | Done when |
|---|---|---|
| 0. Setup | Repo, `pyproject.toml`, pre-commit (ruff), pytest, configs skeleton | `pytest` runs; `python -m lpdp --help` works |
| 1. Data | Download + preprocess NYC and Gowalla subset; splits | `dataset_stats.json` written; split sizes match config; tests for filtering rules |
| 2. Mechanisms | M1, M2, M3, M7 as pure functions on DataFrames | Unit tests: geo-ind mean displacement ≈ 2/ε; k-anon cells all have ≥ k users; cloaked points fall in their cell; M7 noise scale = c/ε |
| 3. Tasks | GRU next-POI (M0), hotspot pipeline | M0 Acc@10 is clearly above a most-popular-POI baseline (report both); hotspot F1(raw vs raw) = 1.0 |
| 4. Private training | M4 DP-SGD, M5 FedAvg, M6 DP-FedAvg + accounting | Reported ε matches config target (M4) / accountant (M6); FedAvg utility within reasonable range of M0 (report the gap, whatever it is) |
| 5. Attacks | A1, A2 | On M0, A2 AUC > 0.5 and A1 succeeds well above chance; sanity: A2 AUC ≈ 0.5 for an untrained model |
| 6. Sweeps | `run_sweep.py` over Section 5 grid × 2 datasets × 3 seeds | `results/runs.csv` complete, no missing cells that should apply |
| 7. Figures | `make_figures.py` → Fig. 5–9 PNG + PDF; Fig. 1–4 drawn as diagrams | All 9 figures exist; abstract placeholders filled from `runs.csv` |
| 8. Compliance + recommender | `compliance/evaluate.py` fills status matrix; rules for `/recommend` | Fig. 9 generated from data, not by hand |
| 9. App | Supabase migration, sync script, FastAPI, React pages | Dashboard shows the same numbers as `runs.csv`; no personal data in Supabase |
| 10. Paper | Fill Sections 5–8 of the paper outline with results | Every number traceable to a run ID |

Approximate run count for the MVP grid: 25 next-POI configs × 2 datasets × 3 seeds = 150 small GRU trainings, plus cheap hotspot runs. Feasible on Colab.

---

## 11. Figures (from the topic brief) and how each is produced

| Fig. | Caption | Source |
|---|---|---|
| 1 | Framework overview | Diagram (draw.io / Mermaid): data → mechanisms → tasks → attacks → compliance |
| 2 | Threat model | Diagram: attacker knowledge for A1 (p known points) and A2 (model access) |
| 3 | Mechanism placement | Diagram: client (M1, M3), curator (M2, M7), model (M4–M6) |
| 4 | Federated setup | Diagram: clients, clipping, server noise, aggregation |
| 5 | Privacy-utility curves | `make_figures.py`: Acc@10 and hotspot F1 vs each mechanism's parameter |
| 6 | Attack success | A1 top-1 and A2 AUC vs parameter |
| 7 | Pareto front | Utility vs attack success, all mechanisms, non-dominated points highlighted |
| 8 | Spatial noise example | Map of raw vs M1/M3 perturbed points (same as dashboard demo) |
| 9 | DPDP mapping | Obligation × mechanism status matrix |

---

## 12. Limitations to state in the paper
- Legal mapping is indicative and not a legal opinion.
- Datasets are US / global social check-in data, not Indian data.
- Federation is simulated (users as clients on one machine).
- DP-SGD epsilon is example-level; user-level only via a bound. This is why attack success is the common axis.
- Erasure from trained models requires retraining; machine unlearning is out of scope.

## 13. Future work (from the brief)
Real-time social sensing; fairness of utility and attack risk across user groups (e.g., highly active vs occasional users, since outliers are more exposed [12]).
