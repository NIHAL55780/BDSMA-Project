"""M3: geo-indistinguishability via the planar Laplace mechanism."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import lambertw

from lpdp.geo import LocalCRS, add_latlon_from_xy
from lpdp.mechanisms.cloaking import snap_to_pois


def planar_laplace_sample(
    n: int,
    epsilon_per_meter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n, 2) displacements in metres.

    θ ~ U[0, 2π); p ~ U[0, 1); r = −(1/ε) (W₋₁((p−1)/e) + 1).
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    p = rng.uniform(0.0, 1.0, size=n)
    # Numerical guard: p=0 → (p-1)/e = -1/e, Lambert W_{-1}(-1/e) = -1, r = 0.
    arg = np.clip((p - 1.0) / np.e, -1.0 / np.e + 1e-15, -1e-15)
    w = lambertw(arg, k=-1).real
    radius = -(1.0 / epsilon_per_meter) * (w + 1.0)
    dx = radius * np.cos(theta)
    dy = radius * np.sin(theta)
    return np.column_stack([dx, dy])


def geo_indistinguishability(
    checkins: pd.DataFrame,
    pois: pd.DataFrame,
    eps_per_km: float,
    rng: np.random.Generator,
    crs: LocalCRS | None = None,
) -> pd.DataFrame:
    """Perturb (x, y) with planar Laplace, then snap to the nearest public POI."""
    out = checkins.copy()
    out["original_poi_id"] = out["poi_id"]
    eps_m = float(eps_per_km) / 1000.0
    dxy = planar_laplace_sample(len(out), eps_m, rng)
    out["x"] = out["x"].to_numpy() + dxy[:, 0]
    out["y"] = out["y"].to_numpy() + dxy[:, 1]
    if crs is not None:
        out = add_latlon_from_xy(out, crs)
    new_poi = snap_to_pois(out["x"].to_numpy(), out["y"].to_numpy(), pois)
    poi_xy = pois.set_index("poi_id")[["x", "y", "lat", "lon"]]
    out["poi_id"] = new_poi
    out["x"] = poi_xy.loc[out["poi_id"], "x"].to_numpy()
    out["y"] = poi_xy.loc[out["poi_id"], "y"].to_numpy()
    out["lat"] = poi_xy.loc[out["poi_id"], "lat"].to_numpy()
    out["lon"] = poi_xy.loc[out["poi_id"], "lon"].to_numpy()
    return out
