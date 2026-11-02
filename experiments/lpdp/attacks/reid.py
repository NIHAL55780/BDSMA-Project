"""A1: spatial-temporal re-identification on released check-ins."""

from __future__ import annotations

import numpy as np
import pandas as pd


def unicity(checkins: pd.DataFrame, p: int, rng: np.random.Generator, d_m: float, dt_hours: float) -> float:
    """Fraction of users uniquely identified by p of their own points (raw unicity)."""
    users = checkins["user_id"].unique()
    if len(users) == 0:
        return 0.0
    unique = 0
    sampled = 0
    dt_ns = int(dt_hours * 3600 * 1_000_000_000)
    for uid in users:
        pts = checkins[checkins["user_id"] == uid]
        if len(pts) < p:
            continue
        sampled += 1
        probe = pts.sample(n=p, random_state=int(rng.integers(0, 1_000_000_000)))
        scores = _scores(probe, checkins, d_m, dt_ns)
        top = max(scores.values()) if scores else 0
        winners = [u for u, s in scores.items() if s == top]
        if winners == [uid]:
            unique += 1
    return unique / max(sampled, 1)


def _scores(
    probe: pd.DataFrame,
    released: pd.DataFrame,
    d_m: float,
    dt_ns: int,
) -> dict[int, int]:
    scores: dict[int, int] = {}
    rel_x = released["x"].to_numpy()
    rel_y = released["y"].to_numpy()
    rel_t = pd.to_datetime(released["ts"], utc=True).astype("int64").to_numpy()
    rel_u = released["user_id"].to_numpy()
    for _, row in probe.iterrows():
        dx = rel_x - float(row["x"])
        dy = rel_y - float(row["y"])
        dist = np.hypot(dx, dy)
        probe_t = pd.Timestamp(row["ts"])
        if probe_t.tzinfo is None:
            probe_t = probe_t.tz_localize("UTC")
        tdelta = np.abs(rel_t - int(probe_t.value))
        hit = (dist <= d_m) & (tdelta <= dt_ns)
        for u in np.unique(rel_u[hit]):
            scores[int(u)] = scores.get(int(u), 0) + 1
    return scores


def reidentification(
    raw: pd.DataFrame,
    released: pd.DataFrame,
    p_list: list[int],
    n_targets: int,
    d_m: float,
    dt_hours: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Top-1 / top-5 re-id rate. Attacker holds p raw points of each target."""
    users = raw.groupby("user_id").size()
    eligible = users[users >= max(p_list)].index.to_numpy()
    if len(eligible) == 0:
        return {f"reid_top1_p{p}": 0.0 for p in p_list} | {f"reid_top5_p{p}": 0.0 for p in p_list}
    n = min(n_targets, len(eligible))
    targets = rng.choice(eligible, size=n, replace=False)
    dt_ns = int(dt_hours * 3600 * 1_000_000_000)
    out: dict[str, float] = {}
    for p in p_list:
        top1 = 0
        top5 = 0
        used = 0
        for uid in targets:
            pts = raw[raw["user_id"] == uid]
            if len(pts) < p:
                continue
            probe = pts.sample(n=p, random_state=int(rng.integers(0, 1_000_000_000)))
            scores = _scores(probe, released, d_m, dt_ns)
            used += 1
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top_ids = [u for u, _ in ranked[:5]]
            if top_ids and top_ids[0] == uid:
                top1 += 1
            if uid in top_ids:
                top5 += 1
        out[f"reid_top1_p{p}"] = top1 / max(used, 1)
        out[f"reid_top5_p{p}"] = top5 / max(used, 1)
    return out
