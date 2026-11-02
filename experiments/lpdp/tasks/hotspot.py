"""T2: hotspot detection on a regular metre grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lpdp.eval.metrics import count_correlation, hotspot_f1
from lpdp.mechanisms.central_dp import central_dp_histogram
from lpdp.mechanisms.cloaking import grid_index


@dataclass
class GridSpec:
    origin: tuple[float, float]
    cell_m: float
    nx: int
    ny: int


def make_grid(checkins: pd.DataFrame, cell_m: float) -> GridSpec:
    xmin, xmax = float(checkins["x"].min()), float(checkins["x"].max())
    ymin, ymax = float(checkins["y"].min()), float(checkins["y"].max())
    pad = cell_m
    origin = (xmin - pad, ymin - pad)
    nx = int(np.ceil((xmax - origin[0]) / cell_m)) + 1
    ny = int(np.ceil((ymax - origin[1]) / cell_m)) + 1
    return GridSpec(origin=origin, cell_m=cell_m, nx=max(nx, 1), ny=max(ny, 1))


def count_grid(checkins: pd.DataFrame, grid: GridSpec) -> np.ndarray:
    ix, iy = grid_index(checkins["x"].to_numpy(), checkins["y"].to_numpy(), grid.cell_m, grid.origin)
    ix = np.clip(ix, 0, grid.nx - 1)
    iy = np.clip(iy, 0, grid.ny - 1)
    counts = np.zeros((grid.nx, grid.ny), dtype=float)
    np.add.at(counts, (ix, iy), 1.0)
    return counts


def top_cells(counts: np.ndarray, top_pct: float) -> set[tuple[int, int]]:
    flat = counts.ravel()
    n = max(1, int(np.ceil(flat.size * top_pct)))
    # Ignore empty grids.
    if flat.max() <= 0:
        return set()
    thresh = np.partition(flat, -n)[-n]
    ix, iy = np.where(counts >= thresh)
    cells = list(zip(ix.tolist(), iy.tolist(), strict=False))
    cells.sort(key=lambda c: counts[c[0], c[1]], reverse=True)
    return set(cells[:n])


def compare_hotspots(true_counts: np.ndarray, pred_counts: np.ndarray, top_pct: float) -> dict[str, float]:
    true_set = top_cells(true_counts, top_pct)
    pred_set = top_cells(pred_counts, top_pct)
    rho, mae = count_correlation(true_counts, pred_counts)
    return {
        "hotspot_f1": hotspot_f1(true_set, pred_set),
        "spearman": rho,
        "count_mae": mae,
        "n_true_hotspots": float(len(true_set)),
        "n_pred_hotspots": float(len(pred_set)),
    }


def run_hotspot(
    raw: pd.DataFrame,
    released: pd.DataFrame | None,
    cfg: dict,
    mechanism: str,
    param_value: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    cell_m = float(cfg["hotspot"]["cell_m"])
    top_pct = float(cfg["hotspot"]["top_pct"])
    grid = make_grid(raw, cell_m)
    true_counts = count_grid(raw, grid)
    if mechanism.upper() == "M7":
        pred = central_dp_histogram(
            raw,
            cell_m=grid.cell_m,
            origin=grid.origin,
            user_cap=int(cfg["hotspot"]["user_cap"]),
            epsilon=float(param_value),
            rng=rng,
            grid_shape=(grid.nx, grid.ny),
        )
    else:
        pred = count_grid(released if released is not None else raw, grid)
    return compare_hotspots(true_counts, pred, top_pct)
