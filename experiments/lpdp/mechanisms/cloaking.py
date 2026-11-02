"""M1: spatial cloaking — replace each check-in by a grid-cell representative POI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def grid_index(x: np.ndarray, y: np.ndarray, cell_m: float, origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    ox, oy = origin
    ix = np.floor((x - ox) / cell_m).astype(int)
    iy = np.floor((y - oy) / cell_m).astype(int)
    return ix, iy


def cell_centroid(
    ix: np.ndarray, iy: np.ndarray, cell_m: float, origin: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    ox, oy = origin
    cx = ox + (ix + 0.5) * cell_m
    cy = oy + (iy + 0.5) * cell_m
    return cx, cy


def snap_to_pois(x: np.ndarray, y: np.ndarray, pois: pd.DataFrame) -> np.ndarray:
    tree = cKDTree(pois[["x", "y"]].to_numpy())
    _, nn = tree.query(np.column_stack([x, y]), k=1)
    return pois["poi_id"].to_numpy()[nn]


def spatial_cloak(
    checkins: pd.DataFrame,
    pois: pd.DataFrame,
    cell_m: float,
    origin: tuple[float, float] | None = None,
) -> pd.DataFrame:
    out = checkins.copy()
    out["original_poi_id"] = out["poi_id"]
    origin = origin or (float(out["x"].min()), float(out["y"].min()))
    ix, iy = grid_index(out["x"].to_numpy(), out["y"].to_numpy(), cell_m, origin)
    cx, cy = cell_centroid(ix, iy, cell_m, origin)
    out["cell_ix"] = ix
    out["cell_iy"] = iy
    out["cell_cx"] = cx
    out["cell_cy"] = cy
    # Cloaked point is the cell centroid; representative POI is nearest to that centroid.
    new_poi = snap_to_pois(cx, cy, pois)
    poi_xy = pois.set_index("poi_id")[["x", "y", "lat", "lon"]]
    out["poi_id"] = new_poi
    out["x"] = poi_xy.loc[out["poi_id"], "x"].to_numpy()
    out["y"] = poi_xy.loc[out["poi_id"], "y"].to_numpy()
    out["lat"] = poi_xy.loc[out["poi_id"], "lat"].to_numpy()
    out["lon"] = poi_xy.loc[out["poi_id"], "lon"].to_numpy()
    return out
