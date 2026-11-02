"""Apply a release-time mechanism (M0–M3) to a check-in table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpdp.geo import LocalCRS
from lpdp.mechanisms.cloaking import spatial_cloak
from lpdp.mechanisms.geoind import geo_indistinguishability
from lpdp.mechanisms.kanon import k_anonymity_cloak


def apply_release_mechanism(
    checkins: pd.DataFrame,
    pois: pd.DataFrame,
    mechanism: str,
    param_value: float,
    rng: np.random.Generator,
    crs: LocalCRS | None = None,
) -> pd.DataFrame:
    mech = mechanism.upper()
    if mech == "M0":
        out = checkins.copy()
        out["original_poi_id"] = out["poi_id"]
        return out
    if mech == "M1":
        return spatial_cloak(checkins, pois, cell_m=float(param_value))
    if mech == "M2":
        return k_anonymity_cloak(checkins, pois, k=int(param_value))
    if mech == "M3":
        return geo_indistinguishability(
            checkins, pois, eps_per_km=float(param_value), rng=rng, crs=crs
        )
    raise ValueError(f"{mechanism} is not a release mechanism")
