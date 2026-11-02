"""Ranking and hotspot metrics."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def ranking_metrics(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """logits [B, N], targets [B] — POI 0 is padding and is never a target."""
    # Mask padding index 0
    scores = logits.clone()
    scores[:, 0] = -1e9
    order = scores.argsort(dim=-1, descending=True)
    # rank of target (1-based)
    match = order.eq(targets.unsqueeze(1))
    # first True along dim 1
    ranks = match.float().argmax(dim=1) + 1
    none = ~match.any(dim=1)
    ranks = ranks.masked_fill(none, scores.size(1))
    ranks_np = ranks.cpu().numpy().astype(float)
    out = {f"acc{k}": float((ranks_np <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / ranks_np).mean())
    return out


def hotspot_f1(true_cells: set[tuple[int, int]], pred_cells: set[tuple[int, int]]) -> float:
    if not true_cells and not pred_cells:
        return 1.0
    tp = len(true_cells & pred_cells)
    prec = tp / len(pred_cells) if pred_cells else 0.0
    rec = tp / len(true_cells) if true_cells else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def count_correlation(true_counts: np.ndarray, pred_counts: np.ndarray) -> tuple[float, float]:
    t = true_counts.ravel()
    p = pred_counts.ravel()
    if t.std() == 0 or p.std() == 0:
        return 0.0, float(np.mean(np.abs(t - p)))
    rho, _ = spearmanr(t, p)
    mae = float(np.mean(np.abs(t - p)))
    return float(rho), mae


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float = 0.01) -> float:
    """scores: higher → more likely member. y_true 1=member, 0=non-member."""
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])
