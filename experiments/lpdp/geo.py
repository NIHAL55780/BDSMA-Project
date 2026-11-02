"""Coordinate projection helpers (WGS84 ↔ local UTM metres)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Transformer


def utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


@dataclass
class LocalCRS:
    epsg: str
    to_xy: Transformer
    to_ll: Transformer

    @classmethod
    def from_bbox(cls, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> LocalCRS:
        lat = (lat_min + lat_max) / 2
        lon = (lon_min + lon_max) / 2
        epsg = utm_epsg(lat, lon)
        return cls(
            epsg=epsg,
            to_xy=Transformer.from_crs("EPSG:4326", epsg, always_xy=True),
            to_ll=Transformer.from_crs(epsg, "EPSG:4326", always_xy=True),
        )

    def project(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.to_xy.transform(lon, lat)
        return np.asarray(x), np.asarray(y)

    def unproject(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self.to_ll.transform(x, y)
        return np.asarray(lon), np.asarray(lat)


def add_xy(df: pd.DataFrame, crs: LocalCRS, lon_col: str = "lon", lat_col: str = "lat") -> pd.DataFrame:
    out = df.copy()
    x, y = crs.project(out[lon_col].to_numpy(), out[lat_col].to_numpy())
    out["x"] = x
    out["y"] = y
    return out


def add_latlon_from_xy(df: pd.DataFrame, crs: LocalCRS) -> pd.DataFrame:
    out = df.copy()
    lon, lat = crs.unproject(out["x"].to_numpy(), out["y"].to_numpy())
    out["lon"] = lon
    out["lat"] = lat
    return out
