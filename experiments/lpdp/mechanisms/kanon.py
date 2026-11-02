"""M2: k-anonymity cloaking via an adaptive quadtree."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lpdp.mechanisms.cloaking import snap_to_pois


@dataclass
class Cell:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    indices: np.ndarray


def _n_users(checkins: pd.DataFrame, indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(checkins.iloc[indices]["user_id"].nunique())


def _quadrants(cell: Cell, checkins: pd.DataFrame) -> list[Cell]:
    xmid = (cell.xmin + cell.xmax) / 2
    ymid = (cell.ymin + cell.ymax) / 2
    xy = checkins[["x", "y"]].to_numpy()
    pts = xy[cell.indices]
    boxes = [
        (cell.xmin, xmid, cell.ymin, ymid),
        (xmid, cell.xmax, cell.ymin, ymid),
        (cell.xmin, xmid, ymid, cell.ymax),
        (xmid, cell.xmax, ymid, cell.ymax),
    ]
    children = []
    for xmin, xmax, ymin, ymax in boxes:
        mask = (
            (pts[:, 0] >= xmin)
            & (pts[:, 0] < xmax)
            & (pts[:, 1] >= ymin)
            & (pts[:, 1] < ymax)
        )
        children.append(
            Cell(xmin, xmax, ymin, ymax, cell.indices[mask])
        )
    return children


def build_kanon_cells(checkins: pd.DataFrame, k: int) -> list[Cell]:
    xmin, xmax = float(checkins["x"].min()), float(checkins["x"].max()) + 1e-6
    ymin, ymax = float(checkins["y"].min()), float(checkins["y"].max()) + 1e-6
    root = Cell(xmin, xmax, ymin, ymax, np.arange(len(checkins)))
    leaves: list[Cell] = []
    stack = [root]
    while stack:
        cell = stack.pop()
        if _n_users(checkins, cell.indices) < k or (cell.xmax - cell.xmin) < 1.0:
            leaves.append(cell)
            continue
        children = _quadrants(cell, checkins)
        nonempty = [c for c in children if len(c.indices) > 0]
        if nonempty and all(_n_users(checkins, c.indices) >= k for c in nonempty):
            stack.extend(nonempty)
        else:
            leaves.append(cell)
    return leaves


def k_anonymity_cloak(checkins: pd.DataFrame, pois: pd.DataFrame, k: int) -> pd.DataFrame:
    out = checkins.copy().reset_index(drop=True)
    out["original_poi_id"] = out["poi_id"]
    leaves = build_kanon_cells(out, k)
    n_users = np.zeros(len(out), dtype=int)
    cx = np.zeros(len(out))
    cy = np.zeros(len(out))
    cell_id = np.zeros(len(out), dtype=int)
    for i, cell in enumerate(leaves):
        if len(cell.indices) == 0:
            continue
        n_u = _n_users(out, cell.indices)
        n_users[cell.indices] = n_u
        cell_cx = (cell.xmin + cell.xmax) / 2
        cell_cy = (cell.ymin + cell.ymax) / 2
        cx[cell.indices] = cell_cx
        cy[cell.indices] = cell_cy
        cell_id[cell.indices] = i
    out["k_cell_users"] = n_users
    out["cell_id"] = cell_id
    out["cell_cx"] = cx
    out["cell_cy"] = cy
    new_poi = snap_to_pois(cx, cy, pois)
    poi_xy = pois.set_index("poi_id")[["x", "y", "lat", "lon"]]
    out["poi_id"] = new_poi
    out["x"] = poi_xy.loc[out["poi_id"], "x"].to_numpy()
    out["y"] = poi_xy.loc[out["poi_id"], "y"].to_numpy()
    out["lat"] = poi_xy.loc[out["poi_id"], "lat"].to_numpy()
    out["lon"] = poi_xy.loc[out["poi_id"], "lon"].to_numpy()
    return out
