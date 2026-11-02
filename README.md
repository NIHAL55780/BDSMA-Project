# Privacy–utility trade-off in social location analytics under India’s DPDP Act

Course project for **Big Data and Social Media Analytics** (Paper 6). Companion to `Paper6_Location_Privacy_DPDP.docx.md` and `Project_Implementation_Plan.md`.

The research artefact is a **Python experiment pipeline**: six privacy mechanisms, two tasks, two attacks, a DPDP checklist. A FastAPI + React dashboard only *displays* the numbers in `results/`. No number in a figure or table is invented — it comes from a logged run.

## What is implemented

| Layer | Contents |
|---|---|
| Data | Foursquare NYC, Gowalla (city bbox), synthetic demo. Filter, UTM projection, 24h sessions, 70/30 user split, 80/10/10 time split, consent + child filters |
| Mechanisms | M0 raw · M1 grid cloaking · M2 k-anonymity quadtree · M3 planar Laplace geo-ind · M4 DP-SGD · M5 FedAvg · M6 DP-FedAvg · M7 Laplace histogram |
| Tasks | Next-POI GRU (Acc@1/5/10, MRR) and 500 m hotspot F1 |
| Attacks | A1 re-identification (p ∈ {2,4,8}) · A2 loss-threshold membership inference |
| Compliance | `checklist.yaml` → obligation × mechanism matrix (Fig. 9) |
| App | FastAPI read API + React dashboard (charts, Leaflet map, checklist, recommender) |

Epsilon is **not** comparable across mechanisms. Pareto plots use **empirical attack success** as the common privacy axis.

## Setup

Python 3.11+ (tested on 3.12). From the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev,app]"
```

```bash
cd frontend
npm install
```

## Quick demo (CPU, synthetic data)

No external download. Tiny GRU, compact sweep, real metrics:

```bash
python -m lpdp --help
python -m lpdp demo --seed 0
pytest
```

Then:

```bash
uvicorn backend.app.main:app --reload
```

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. The dashboard reads `results/runs.csv`.

## Real datasets (paper tables)

Non-commercial research data. Files land in `data/raw/` (gitignored). Never upload raw or per-user data to Supabase.

```bash
python -m lpdp preprocess --dataset foursquare_nyc --seed 0
python -m lpdp preprocess --dataset gowalla --seed 0
```

- Foursquare NYC: Dingqi Yang TSMC 2014 (`dataset_TSMC2014_NYC.txt`)
- Gowalla: SNAP `loc-gowalla_totalCheckins.txt.gz`, clipped to the Austin bbox in `experiments/configs/gowalla.yaml`

Edit `experiments/configs/sweep.yaml` to the full MVP grid (commented in the file), then:

```bash
python -m lpdp sweep --dataset foursquare_nyc
python -m lpdp make-figures --dataset foursquare_nyc
python -m lpdp compliance --dataset foursquare_nyc
```

Seeds `0, 1, 2` are the default in `sweep.yaml`. Report mean ± std.

## Config, not code

Hyperparameters live in `experiments/configs/*.yaml`. Attack pass/fail thresholds for the “security safeguards” checklist row are in `experiments/configs/policy.yaml` (project policy, not legal standards).

## Dashboard API

| Method | Path | Role |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/datasets`, `/mechanisms` | catalogues |
| GET | `/runs?dataset=&task=&mechanism=` | metrics averaged over seeds |
| GET | `/pareto?dataset=&task=&utility=&risk=` | points + non-dominated set |
| GET | `/compliance?dataset=` | checklist matrix |
| POST | `/perturb` | public landmarks only (not user check-ins) |
| POST | `/recommend` | mechanisms meeting min utility and max risk |

Optional: copy `.env.example` to `.env` and run `python -m experiments.scripts.sync_supabase` to push **aggregates** after applying `supabase/migrations/0001_init.sql`. The app runs from CSV without Supabase.

## Tests (phase acceptance)

```bash
pytest
```

Includes: geo-ind mean displacement ≈ 2/ε (100k samples), k-anon cells, cloaked points in cell, M7 noise scale = c/ε, hotspot F1(raw, raw) = 1, A1 above chance on raw data, A2 ≈ 0.5 on an untrained model, schema minimisation, consent/erasure/child filters.

## Layout

```
experiments/lpdp/     pipeline package
experiments/configs/  YAML
experiments/scripts/  sweep, figures, supabase sync
experiments/tests/
backend/app/         FastAPI
frontend/             React + Vite + TypeScript
supabase/migrations/
results/              runs.csv (the only numbers that count)
figures/              Fig. 1–9 PNG + PDF
```

## Limitations (also in the paper)

Legal mapping is indicative. Datasets are US/global check-ins, not Indian data. Federation is simulated on one machine. DP-SGD ε is example-level; user-level is a bound. Erasure from trained models requires retraining.
