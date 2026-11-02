"""CSV-backed store. Optional Supabase is a sync target, not required to run the app."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from lpdp.config import REPO_ROOT, load_config, resolve_path
from lpdp.eval.pareto import non_dominated

METRIC_COLS = [
    "acc1",
    "acc5",
    "acc10",
    "mrr",
    "popular_acc10",
    "hotspot_f1",
    "spearman",
    "count_mae",
    "reid_top1_p2",
    "reid_top1_p4",
    "reid_top1_p8",
    "mia_auc",
    "mia_tpr_at_1fpr",
    "comm_bytes_per_round",
    "train_seconds",
    "epsilon_reported",
]

MECHANISMS = [
    {"id": "M0", "name": "None (raw)", "acts_at": "none", "param_name": None},
    {"id": "M1", "name": "Spatial cloaking", "acts_at": "client", "param_name": "cell_m"},
    {"id": "M2", "name": "k-anonymity cloaking", "acts_at": "curator", "param_name": "k"},
    {"id": "M3", "name": "Geo-indistinguishability", "acts_at": "client", "param_name": "eps_per_km"},
    {"id": "M4", "name": "DP-SGD", "acts_at": "model", "param_name": "epsilon"},
    {"id": "M5", "name": "FedAvg", "acts_at": "model", "param_name": None},
    {"id": "M6", "name": "DP-FedAvg", "acts_at": "model", "param_name": "z"},
    {"id": "M7", "name": "Central DP histogram", "acts_at": "aggregate", "param_name": "epsilon"},
]


def runs_csv() -> Path:
    return REPO_ROOT / "results" / "runs.csv"


def load_runs() -> pd.DataFrame:
    path = runs_csv()
    if not path.exists():
        return pd.DataFrame(columns=["dataset", "task", "mechanism", "param_value", "seed", *METRIC_COLS])
    return pd.read_csv(path)


def list_datasets() -> list[dict]:
    out = []
    for ds in ("synthetic", "foursquare_nyc", "gowalla"):
        cfg = load_config(ds)
        stats_path = resolve_path(cfg, "processed") / ds / "dataset_stats.json"
        item = {"id": ds, "name": ds.replace("_", " ").title(), "n_users": None, "n_pois": None, "n_checkins": None}
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            item.update(stats.get("after", {}))
        out.append(item)
    return out


def aggregated_runs(dataset: str | None, task: str | None, mechanism: str | None) -> list[dict]:
    df = load_runs()
    if df.empty:
        return []
    if dataset:
        df = df[df["dataset"] == dataset]
    if task:
        df = df[df["task"] == task]
    if mechanism:
        df = df[df["mechanism"] == mechanism]
    keys = ["dataset", "task", "mechanism", "param_name", "param_value"]
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        rec = dict(zip(keys, key, strict=False))
        rec["n_seeds"] = int(g["seed"].nunique())
        metrics = {}
        for col in METRIC_COLS:
            if col not in g.columns:
                continue
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if s.empty:
                continue
            metrics[col] = {"mean": float(s.mean()), "std": float(s.std(ddof=0))}
        rec["metrics"] = metrics
        rows.append(rec)
    return rows


def pareto(dataset: str, task: str, utility: str, risk: str) -> dict:
    df = load_runs()
    df = df[(df["dataset"] == dataset) & (df["task"] == task)]
    if df.empty:
        return {"points": [], "front": []}
    agg = (
        df.groupby(["mechanism", "param_value"], as_index=False)
        .agg(utility=(utility, "mean"), risk=(risk, "mean"))
        .dropna()
    )
    agg = agg.rename(columns={"utility": utility, "risk": risk})
    front = non_dominated(agg, utility, risk)
    return {
        "points": agg.to_dict(orient="records"),
        "front": front.to_dict(orient="records"),
    }


def compliance(dataset: str) -> dict:
    path = REPO_ROOT / "results" / "compliance_status.csv"
    from lpdp.compliance.evaluate import load_checklist

    items = load_checklist()
    if not path.exists():
        return {"items": items, "status": []}
    df = pd.read_csv(path)
    df = df[df["dataset_id"] == dataset]
    return {"items": items, "status": df.to_dict(orient="records")}


def recommend(task: str, max_risk: float, min_utility: float) -> list[dict]:
    df = load_runs()
    if df.empty:
        return []
    df = df[df["task"] == task]
    util_col = "acc10" if task == "next_poi" else "hotspot_f1"
    risk_col = "mia_auc" if task == "next_poi" else "reid_top1_p4"
    agg = (
        df.groupby(["dataset", "mechanism", "param_name", "param_value"], as_index=False)
        .agg(utility=(util_col, "mean"), risk=(risk_col, "mean"))
        .dropna()
    )
    ok = agg[(agg["utility"] >= min_utility) & (agg["risk"] <= max_risk)]
    ok = ok.sort_values(["utility", "risk"], ascending=[False, True])
    return ok.to_dict(orient="records")
