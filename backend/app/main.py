from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from backend.app.store import (  # noqa: E402
    MECHANISMS,
    aggregated_runs,
    compliance,
    list_datasets,
    pareto,
    recommend,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lpdp.api")

app = FastAPI(
    title="Location Privacy × DPDP",
    description="Read API over experiment aggregates. No personal data is stored or returned.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit(request: Request, call_next):
    response = await call_next(request)
    log.info("audit %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def datasets():
    return list_datasets()


@app.get("/mechanisms")
def mechanisms():
    return MECHANISMS


@app.get("/runs")
def runs(dataset: str | None = None, task: str | None = None, mechanism: str | None = None):
    return aggregated_runs(dataset, task, mechanism)


@app.get("/pareto")
def pareto_ep(
    dataset: str = "synthetic",
    task: str = "next_poi",
    utility: str = "acc10",
    risk: str = "mia_auc",
):
    return pareto(dataset, task, utility, risk)


@app.get("/compliance")
def compliance_ep(dataset: str = "synthetic"):
    return compliance(dataset)


class PerturbBody(BaseModel):
    mechanism: str = "M3"
    param: float = 2.0
    n_points: int = Field(default=20, ge=1, le=80)


@app.post("/perturb")
def perturb(body: PerturbBody):
    """Perturb a fixed public sample of landmark coordinates — never user check-ins."""
    import numpy as np
    import pandas as pd

    from lpdp.config import REPO_ROOT
    from lpdp.geo import LocalCRS, add_xy
    from lpdp.mechanisms.cloaking import spatial_cloak
    from lpdp.mechanisms.geoind import geo_indistinguishability

    path = REPO_ROOT / "data" / "demo" / "nyc_landmarks.csv"
    pois = pd.read_csv(path)
    pois = pois.head(body.n_points).copy()
    pois["poi_id"] = np.arange(1, len(pois) + 1)
    pois["user_id"] = 1
    bbox = {
        "lat_min": float(pois["lat"].min()),
        "lat_max": float(pois["lat"].max()),
        "lon_min": float(pois["lon"].min()),
        "lon_max": float(pois["lon"].max()),
    }
    crs = LocalCRS.from_bbox(**bbox)
    pois = add_xy(pois, crs)
    rng = np.random.default_rng(0)
    mech = body.mechanism.upper()
    if mech == "M1":
        out = spatial_cloak(pois, pois, cell_m=max(body.param, 50.0))
    elif mech == "M3":
        out = geo_indistinguishability(pois, pois, eps_per_km=body.param, rng=rng, crs=crs)
    else:
        out = pois.copy()
        out["original_poi_id"] = out["poi_id"]
    raw = pois[["name", "lat", "lon"]].to_dict(orient="records")
    pert = out[["lat", "lon"]].copy()
    pert["name"] = pois["name"].to_numpy()
    return {"raw": raw, "perturbed": pert.to_dict(orient="records"), "mechanism": mech, "param": body.param}


class RecommendBody(BaseModel):
    task: str = "next_poi"
    max_risk: float = 0.55
    min_utility: float = 0.1


@app.post("/recommend")
def recommend_ep(body: RecommendBody):
    return recommend(body.task, body.max_risk, body.min_utility)
