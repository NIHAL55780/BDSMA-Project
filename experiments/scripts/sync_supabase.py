"""Upload aggregate run metrics (never raw check-ins) to Supabase."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from lpdp.compliance.evaluate import MECHANISM_META, load_checklist  # noqa: E402
from lpdp.config import load_config, resolve_path  # noqa: E402


def _client():
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(ROOT / ".env")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    return create_client(url, key)


def sync(dataset: str = "synthetic") -> None:
    cfg = load_config(dataset)
    results = resolve_path(cfg, "results")
    runs_path = results / "runs.csv"
    if not runs_path.exists():
        raise SystemExit(f"Missing {runs_path}")
    sb = _client()
    runs = pd.read_csv(runs_path)
    runs = runs[runs["dataset"] == dataset]

    stats_path = resolve_path(cfg, "processed") / dataset / "dataset_stats.json"
    n_users = n_pois = n_checkins = None
    if stats_path.exists():
        import json

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        n_users = stats.get("after", {}).get("n_users")
        n_pois = stats.get("after", {}).get("n_pois")
        n_checkins = stats.get("after", {}).get("n_checkins")
    sb.table("datasets").upsert(
        {"id": dataset, "name": dataset, "n_users": n_users, "n_pois": n_pois, "n_checkins": n_checkins}
    ).execute()
    for mid, meta in MECHANISM_META.items():
        sb.table("mechanisms").upsert({"id": mid, "name": meta["name"], "acts_at": meta["acts_at"]}).execute()
    for item in load_checklist():
        sb.table("compliance_items").upsert(
            {
                "id": item["id"],
                "obligation": item["obligation"],
                "act_reference": item["act_reference"],
                "control": item["control"],
                "test": item["test"],
            }
        ).execute()

    metric_names = [
        "acc1",
        "acc5",
        "acc10",
        "mrr",
        "hotspot_f1",
        "reid_top1_p2",
        "reid_top1_p4",
        "reid_top1_p8",
        "mia_auc",
        "mia_tpr_at_1fpr",
        "comm_bytes_per_round",
        "train_seconds",
    ]
    for _, row in runs.iterrows():
        payload = {
            "dataset_id": row["dataset"],
            "mechanism_id": row["mechanism"],
            "task": row["task"],
            "param_value": None if pd.isna(row.get("param_value")) else float(row["param_value"]),
            "epsilon_reported": None if pd.isna(row.get("epsilon_reported")) else float(row["epsilon_reported"]),
            "seed": int(row["seed"]),
        }
        inserted = sb.table("runs").insert(payload).execute()
        run_id = inserted.data[0]["id"]
        metrics = []
        for name in metric_names:
            if name in row and pd.notna(row[name]):
                metrics.append({"run_id": run_id, "name": name, "value": float(row[name])})
        if metrics:
            sb.table("metrics").upsert(metrics).execute()

    comp_path = results / "compliance_status.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        for _, row in comp.iterrows():
            sb.table("compliance_status").upsert(
                {
                    "item_id": row["item_id"],
                    "mechanism_id": row["mechanism_id"],
                    "dataset_id": row["dataset_id"],
                    "status": row["status"],
                    "evidence": row.get("evidence"),
                }
            ).execute()
    print("Sync complete (aggregates only — no check-ins uploaded).")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    sync(ds)
