"""A2: loss-threshold membership inference on trained next-POI models."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from lpdp.eval.metrics import tpr_at_fpr
from lpdp.train.centralised import per_window_loss
from lpdp.train.windows import Window


def user_mean_loss(model, windows: list[Window]) -> dict[int, float]:
    rows = per_window_loss(model, windows)
    acc: dict[int, list[float]] = defaultdict(list)
    for uid, loss in rows:
        acc[uid].append(loss)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def membership_inference(
    model,
    member_windows: list[Window],
    nonmember_windows: list[Window],
    fpr_target: float = 0.01,
    max_users: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Lower loss → more likely member. Balanced sampling of users."""
    mem_loss = user_mean_loss(model, member_windows)
    non_loss = user_mean_loss(model, nonmember_windows)
    mem_ids = list(mem_loss)
    non_ids = list(non_loss)
    if not mem_ids or not non_ids:
        return {"mia_auc": 0.5, "mia_tpr_at_1fpr": 0.0}
    n = min(len(mem_ids), len(non_ids), max_users or 10**9)
    rng = rng or np.random.default_rng(0)
    mem_ids = rng.choice(mem_ids, size=n, replace=False).tolist()
    non_ids = rng.choice(non_ids, size=n, replace=False).tolist()
    y = np.array([1] * n + [0] * n)
    # score: negative loss so higher = more member-like
    scores = np.array([-mem_loss[u] for u in mem_ids] + [-non_loss[u] for u in non_ids])
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = 0.5
    tpr = tpr_at_fpr(y, scores, fpr_target=fpr_target)
    return {"mia_auc": auc, "mia_tpr_at_1fpr": tpr}
