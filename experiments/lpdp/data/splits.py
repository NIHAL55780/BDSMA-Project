"""User membership split and per-user time split."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_user_split(
    checkins: pd.DataFrame,
    member_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (member_ids, nonmember_ids) stratified by activity quartile."""
    counts = checkins.groupby("user_id").size().rename("n").reset_index()
    counts["bin"] = pd.qcut(counts["n"], q=min(4, counts["n"].nunique()), duplicates="drop")
    rng = np.random.default_rng(seed)
    members: list[int] = []
    nonmembers: list[int] = []
    for _, group in counts.groupby("bin", observed=True):
        ids = group["user_id"].to_numpy()
        rng.shuffle(ids)
        cut = max(1, int(round(len(ids) * member_frac)))
        cut = min(cut, len(ids) - 1) if len(ids) > 1 else len(ids)
        members.extend(ids[:cut].tolist())
        nonmembers.extend(ids[cut:].tolist())
    return np.array(members, dtype=int), np.array(nonmembers, dtype=int)


def assign_time_splits(
    checkins: pd.DataFrame,
    member_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> pd.DataFrame:
    """For members: first 80% train, next 10% val, last 10% test. Non-members: 'holdout'."""
    out = checkins.sort_values(["user_id", "ts"]).copy()
    out["split"] = "holdout"
    member_set = {int(x) for x in member_ids.tolist()}
    for uid, group in out.groupby("user_id"):
        if int(uid) not in member_set:
            continue
        idx = group.index.to_numpy()
        n = len(idx)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac)) if n - n_train >= 2 else 0
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 2 else 0
            n_test = n - n_train - n_val
        out.loc[idx[:n_train], "split"] = "train"
        if n_val:
            out.loc[idx[n_train : n_train + n_val], "split"] = "val"
        if n_test:
            out.loc[idx[n_train + n_val :], "split"] = "test"
    return out
