"""Synthetic check-ins for tests and the CPU demo (not a research dataset)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from lpdp.geo import LocalCRS, add_xy


CATEGORIES = [
    "Coffee Shop",
    "Park",
    "Subway",
    "Restaurant",
    "Bar",
    "Museum",
    "Gym",
    "Grocery",
]


def generate_synthetic(
    n_users: int = 80,
    n_pois: int = 40,
    checkins_per_user: int = 24,
    n_children: int = 4,
    bbox: dict | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bbox = bbox or {
        "lat_min": 40.70,
        "lat_max": 40.80,
        "lon_min": -74.02,
        "lon_max": -73.94,
    }
    rng = np.random.default_rng(seed)
    crs = LocalCRS.from_bbox(**bbox)

    poi_lat = rng.uniform(bbox["lat_min"], bbox["lat_max"], n_pois)
    poi_lon = rng.uniform(bbox["lon_min"], bbox["lon_max"], n_pois)
    pois = pd.DataFrame(
        {
            "poi_raw": [f"P{i}" for i in range(n_pois)],
            "lat": poi_lat,
            "lon": poi_lon,
            "category": rng.choice(CATEGORIES, size=n_pois),
        }
    )
    pois = add_xy(pois, crs)

    home_poi = rng.integers(0, n_pois, size=n_users)
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    rows: list[dict] = []
    for u in range(n_users):
        t = start + timedelta(days=int(rng.integers(0, 20)))
        # Prefer a small set of POIs (home + a few others) so sequences are learnable.
        fav = np.unique(
            np.concatenate(
                [
                    [home_poi[u]],
                    rng.integers(0, n_pois, size=3),
                ]
            )
        )
        for _ in range(checkins_per_user):
            t = t + timedelta(hours=float(rng.uniform(1, 8)))
            poi = int(rng.choice(fav))
            rows.append(
                {
                    "user_raw": f"U{u}",
                    "poi_raw": f"P{poi}",
                    "lat": float(pois.loc[poi, "lat"]),
                    "lon": float(pois.loc[poi, "lon"]),
                    "ts": t,
                    "category": pois.loc[poi, "category"],
                    "is_child": u < n_children,
                }
            )
    checkins = pd.DataFrame(rows)
    return checkins, pois
