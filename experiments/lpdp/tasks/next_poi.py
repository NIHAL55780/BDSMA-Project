"""T1: next-POI prediction — train, evaluate, popular baseline."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from lpdp.fl.dp_fedavg import train_dp_fedavg
from lpdp.fl.fedavg import train_fedavg
from lpdp.models.gru_next_poi import PopularBaseline
from lpdp.train.centralised import evaluate_model, train_centralised
from lpdp.train.dpsgd import train_dpsgd
from lpdp.train.windows import Window, build_windows, cap_windows_per_user


def split_windows(windows: list[Window]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {"train": [], "val": [], "test": [], "holdout": []}
    for w in windows:
        out.setdefault(w.split, []).append(w)
    return out


def by_user(windows: list[Window]) -> dict[int, list[Window]]:
    d: dict[int, list[Window]] = defaultdict(list)
    for w in windows:
        d[w.user_id].append(w)
    return dict(d)


def run_next_poi(
    released: pd.DataFrame,
    cfg: dict,
    mechanism: str,
    param_value: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    seq_len = int(cfg["model"]["seq_len"])
    n_pois = int(released["original_poi_id"].max() if "original_poi_id" in released else released["poi_id"].max())
    windows = build_windows(released, seq_len=seq_len)
    cap = int(cfg["model"]["windows_per_user_cap"])
    windows = cap_windows_per_user(windows, cap, rng)
    parts = split_windows(windows)
    train, val, test = parts["train"], parts["val"], parts["test"]
    holdout = parts.get("holdout", [])

    popular = PopularBaseline()
    popular.fit([w.target for w in train])
    pop_metrics = popular.metrics([w.target for w in test], n_pois=n_pois)

    mech = mechanism.upper()
    comm = None
    eps = None
    model = None
    seconds = 0.0

    if mech in {"M0", "M1", "M2", "M3"}:
        result = train_centralised(n_pois, train, val, cfg)
        model, seconds = result.model, result.seconds
        n_params = result.n_params
    elif mech == "M4":
        result = train_dpsgd(n_pois, train, val, cfg, target_epsilon=float(param_value), seed=seed)
        model, seconds, eps = result.model, result.seconds, result.epsilon
        n_params = result.n_params
    elif mech == "M5":
        result = train_fedavg(n_pois, by_user(train), val, cfg, seed=seed)
        model, seconds, comm = result.model, result.seconds, result.comm_bytes_per_round
        n_params = result.n_params
    elif mech == "M6":
        result = train_dp_fedavg(
            n_pois, by_user(train), val, cfg, noise_multiplier=float(param_value), seed=seed
        )
        model, seconds, comm, eps = result.model, result.seconds, result.comm_bytes_per_round, result.epsilon
        n_params = result.n_params
    else:
        raise ValueError(f"{mechanism} does not apply to next-POI")

    util = evaluate_model(model, test, batch_size=cfg["model"]["batch_size"])
    return {
        "model": model,
        "train_windows": train,
        "val_windows": val,
        "test_windows": test,
        "holdout_windows": holdout,
        "all_windows": windows,
        "acc1": util["acc1"],
        "acc5": util["acc5"],
        "acc10": util["acc10"],
        "mrr": util["mrr"],
        "popular_acc1": pop_metrics["acc1"],
        "popular_acc10": pop_metrics["acc10"],
        "train_seconds": seconds,
        "comm_bytes_per_round": comm,
        "epsilon_reported": eps,
        "n_params": n_params,
        "n_train_windows": len(train),
        "n_test_windows": len(test),
    }
