"""M7: central DP histogram of grid counts (hotspot task only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpdp.mechanisms.cloaking import grid_index


def laplace_scale(user_cap: int, epsilon: float) -> float:
    """Laplace noise scale b = sensitivity / ε = c / ε."""
    return float(user_cap) / float(epsilon)


def cap_user_checkins(checkins: pd.DataFrame, user_cap: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for _, g in checkins.groupby("user_id"):
        if len(g) <= user_cap:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=user_cap, replace=False)
            parts.append(g.loc[idx])
    return pd.concat(parts, ignore_index=True)


def central_dp_histogram(
    checkins: pd.DataFrame,
    cell_m: float,
    origin: tuple[float, float],
    user_cap: int,
    epsilon: float,
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Return noisy count grid. Sensitivity = user_cap, noise ~ Lap(c/ε)."""
    capped = cap_user_checkins(checkins, user_cap, rng)
    ix, iy = grid_index(capped["x"].to_numpy(), capped["y"].to_numpy(), cell_m, origin)
    nx, ny = grid_shape
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    counts = np.zeros((nx, ny), dtype=float)
    np.add.at(counts, (ix, iy), 1.0)
    scale = laplace_scale(user_cap, epsilon)
    noise = rng.laplace(loc=0.0, scale=scale, size=counts.shape)
    return counts + noise
