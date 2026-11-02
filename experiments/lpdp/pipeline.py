"""One experiment run: mechanism × task × seed → a results row."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lpdp.attacks.mia import membership_inference
from lpdp.attacks.reid import reidentification
from lpdp.config import load_config, resolve_path
from lpdp.data.preprocess import load_processed, preprocess_dataset
from lpdp.geo import LocalCRS
from lpdp.mechanisms.dispatch import apply_release_mechanism
from lpdp.tasks.hotspot import run_hotspot
from lpdp.tasks.next_poi import run_next_poi

RUN_COLUMNS = [
    "run_id",
    "dataset",
    "task",
    "mechanism",
    "param_name",
    "param_value",
    "epsilon_reported",
    "seed",
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
    "created_at",
]


def _empty_row() -> dict:
    return {c: None for c in RUN_COLUMNS}


def ensure_processed(dataset: str, seed: int) -> None:
    cfg = load_config(dataset)
    d = resolve_path(cfg, "processed") / dataset
    if not (d / "checkins.parquet").exists():
        preprocess_dataset(dataset, seed=seed)


def run_one(
    dataset: str,
    task: str,
    mechanism: str,
    param_name: str,
    param_value: float,
    seed: int,
) -> dict:
    cfg = load_config(dataset)
    cfg["seed"] = seed
    ensure_processed(dataset, seed)
    checkins, pois, split = load_processed(dataset)
    crs = LocalCRS.from_bbox(**split["bbox"])
    rng = np.random.default_rng(seed)
    mech = mechanism.upper()
    row = _empty_row()
    row.update(
        {
            "run_id": f"{dataset}-{task}-{mech}-{param_name}{param_value}-s{seed}",
            "dataset": dataset,
            "task": task,
            "mechanism": mech,
            "param_name": param_name,
            "param_value": float(param_value),
            "seed": int(seed),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    released = None
    if mech in {"M0", "M1", "M2", "M3"}:
        released = apply_release_mechanism(checkins, pois, mech, param_value, rng, crs=crs)

    atk = cfg["attacks"]

    if task == "hotspot":
        if mech in {"M4", "M5", "M6"}:
            return row  # not applicable — leave metrics empty
        metrics = run_hotspot(checkins, released, cfg, mech, param_value, rng)
        row["hotspot_f1"] = metrics["hotspot_f1"]
        row["spearman"] = metrics["spearman"]
        row["count_mae"] = metrics["count_mae"]
        if released is not None:
            reid = reidentification(
                checkins,
                released,
                p_list=list(atk["reid_p"]),
                n_targets=int(atk["reid_n_targets"]),
                d_m=float(atk["reid_d_m"]),
                dt_hours=float(atk["reid_dt_hours"]),
                rng=rng,
            )
            for k, v in reid.items():
                if k in row:
                    row[k] = v
        if mech == "M7":
            row["epsilon_reported"] = float(param_value)
        return row

    if task != "next_poi":
        raise ValueError(task)

    if mech == "M7":
        return row  # not applicable

    data_for_model = released if released is not None else checkins.assign(original_poi_id=checkins["poi_id"])
    result = run_next_poi(data_for_model, cfg, mech, param_value, seed)
    row["acc1"] = result["acc1"]
    row["acc5"] = result["acc5"]
    row["acc10"] = result["acc10"]
    row["mrr"] = result["mrr"]
    row["popular_acc10"] = result["popular_acc10"]
    row["train_seconds"] = result["train_seconds"]
    row["comm_bytes_per_round"] = result["comm_bytes_per_round"]
    row["epsilon_reported"] = result["epsilon_reported"]
    if mech == "M4":
        row["epsilon_reported"] = float(param_value) if result["epsilon_reported"] is None else result["epsilon_reported"]

    if released is not None:
        reid = reidentification(
            checkins,
            released,
            p_list=list(atk["reid_p"]),
            n_targets=int(atk["reid_n_targets"]),
            d_m=float(atk["reid_d_m"]),
            dt_hours=float(atk["reid_dt_hours"]),
            rng=rng,
        )
        for k, v in reid.items():
            if k in row:
                row[k] = v

    mia = membership_inference(
        result["model"],
        member_windows=result["train_windows"] + result["val_windows"] + result["test_windows"],
        nonmember_windows=result["holdout_windows"],
        fpr_target=float(atk["mia_tpr_fpr"]),
        max_users=int(atk["mia_max_users"]),
        rng=rng,
    )
    row["mia_auc"] = mia["mia_auc"]
    row["mia_tpr_at_1fpr"] = mia["mia_tpr_at_1fpr"]
    return row


def append_run(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        prev = pd.read_csv(csv_path)
        prev = prev[prev["run_id"] != row["run_id"]]
        df = pd.concat([prev, new], ignore_index=True)
    else:
        df = new
    df.to_csv(csv_path, index=False)
