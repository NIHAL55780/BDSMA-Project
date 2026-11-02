from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lpdp.data.preprocess import add_sessions, iterative_filter, reindex
from lpdp.data.synthetic import generate_synthetic
from lpdp.geo import LocalCRS, add_xy


@pytest.fixture
def synthetic_raw():
    checkins, pois = generate_synthetic(
        n_users=40,
        n_pois=20,
        checkins_per_user=16,
        n_children=3,
        seed=0,
    )
    return checkins, pois


@pytest.fixture
def processed_tiny(synthetic_raw):
    checkins, pois = synthetic_raw
    filtered = iterative_filter(checkins, min_user=8, min_poi=3)
    checkins, pois = reindex(filtered, pois)
    bbox = {
        "lat_min": 40.70,
        "lat_max": 40.80,
        "lon_min": -74.02,
        "lon_max": -73.94,
    }
    crs = LocalCRS.from_bbox(**bbox)
    checkins = add_xy(checkins, crs)
    pois = add_xy(pois, crs)
    checkins = add_sessions(checkins, gap_hours=24)
    checkins["split"] = "train"
    n = len(checkins)
    checkins.iloc[int(n * 0.8) : int(n * 0.9), checkins.columns.get_loc("split")] = "val"
    checkins.iloc[int(n * 0.9) :, checkins.columns.get_loc("split")] = "test"
    # Per-user splits are more realistic; overwrite with a proper per-user cut.
    parts = []
    for _, g in checkins.groupby("user_id"):
        g = g.sort_values("ts").copy()
        m = len(g)
        g.iloc[: max(1, int(m * 0.8)), g.columns.get_loc("split")] = "train"
        g.iloc[max(1, int(m * 0.8)) : max(2, int(m * 0.9)), g.columns.get_loc("split")] = "val"
        g.iloc[max(2, int(m * 0.9)) :, g.columns.get_loc("split")] = "test"
        parts.append(g)
    checkins = pd.concat(parts, ignore_index=True)
    return checkins, pois, crs


@pytest.fixture
def rng():
    return np.random.default_rng(0)
