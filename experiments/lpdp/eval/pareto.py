"""Non-dominated (Pareto) front: maximise utility, minimise risk."""

from __future__ import annotations

import pandas as pd


def non_dominated(df: pd.DataFrame, utility: str, risk: str) -> pd.DataFrame:
    pts = df.dropna(subset=[utility, risk]).copy()
    if pts.empty:
        return pts
    u = pts[utility].to_numpy()
    r = pts[risk].to_numpy()
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if (u[j] >= u[i] and r[j] <= r[i]) and (u[j] > u[i] or r[j] < r[i]):
                dominated = True
                break
        keep.append(not dominated)
    out = pts.loc[keep].copy()
    out["pareto"] = True
    return out
