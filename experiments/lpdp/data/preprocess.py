"""Filter, re-index, project, sessionise and split location check-ins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lpdp.config import load_config, resolve_path
from lpdp.data.consent import default_consent_table, filter_by_consent, filter_children
from lpdp.data.download import download_dataset
from lpdp.data.splits import assign_time_splits, stratified_user_split
from lpdp.data.synthetic import generate_synthetic
from lpdp.geo import LocalCRS, add_xy

ALLOWED_COLUMNS = [
    "user_id",
    "poi_id",
    "ts",
    "lat",
    "lon",
    "x",
    "y",
    "session_id",
    "split",
    "is_child",
]


def _parse_foursquare(path: Path, bbox: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "user_raw",
            "poi_raw",
            "category_id",
            "category",
            "lat",
            "lon",
            "tz_offset",
            "ts_raw",
        ],
        encoding="latin-1",
    )
    df["ts"] = pd.to_datetime(df["ts_raw"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "lat", "lon"])
    df = _clip_bbox(df, bbox)
    pois = (
        df.groupby("poi_raw", as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"), category=("category", "first"))
    )
    checkins = df[["user_raw", "poi_raw", "ts", "lat", "lon", "category"]].copy()
    checkins["is_child"] = False
    return checkins, pois


def _parse_gowalla(path: Path, bbox: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_raw", "ts_raw", "lat", "lon", "poi_raw"],
        chunksize=200_000,
    )
    for chunk in reader:
        clipped = _clip_bbox(chunk, bbox)
        if not clipped.empty:
            chunks.append(clipped)
    if not chunks:
        raise ValueError(f"No Gowalla check-ins inside bbox {bbox}")
    df = pd.concat(chunks, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts_raw"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["ts", "lat", "lon"])
    df["category"] = "unknown"
    pois = (
        df.groupby("poi_raw", as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"), category=("category", "first"))
    )
    checkins = df[["user_raw", "poi_raw", "ts", "lat", "lon", "category"]].copy()
    checkins["is_child"] = False
    return checkins, pois


def _clip_bbox(df: pd.DataFrame, bbox: dict) -> pd.DataFrame:
    return df[
        (df["lat"] >= bbox["lat_min"])
        & (df["lat"] <= bbox["lat_max"])
        & (df["lon"] >= bbox["lon_min"])
        & (df["lon"] <= bbox["lon_max"])
    ].copy()


def iterative_filter(
    checkins: pd.DataFrame,
    min_user: int,
    min_poi: int,
) -> pd.DataFrame:
    df = checkins.copy()
    while True:
        user_ok = df.groupby("user_raw").size()
        poi_ok = df.groupby("poi_raw").size()
        keep_u = set(user_ok[user_ok >= min_user].index)
        keep_p = set(poi_ok[poi_ok >= min_poi].index)
        nxt = df[df["user_raw"].isin(keep_u) & df["poi_raw"].isin(keep_p)]
        if len(nxt) == len(df):
            return nxt.reset_index(drop=True)
        df = nxt


def add_sessions(checkins: pd.DataFrame, gap_hours: float) -> pd.DataFrame:
    df = checkins.sort_values(["user_id", "ts"]).copy()
    gap = pd.Timedelta(hours=gap_hours)
    parts = []
    sid = 0
    for _, g in df.groupby("user_id", sort=False):
        g = g.copy()
        deltas = g["ts"].diff()
        new_session = deltas.isna() | (deltas > gap)
        local = new_session.cumsum() - 1
        g["session_id"] = sid + local
        sid = int(g["session_id"].max()) + 1
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def reindex(checkins: pd.DataFrame, pois: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = {u: i + 1 for i, u in enumerate(sorted(checkins["user_raw"].unique(), key=str))}
    poi_ids = {p: i + 1 for i, p in enumerate(sorted(checkins["poi_raw"].unique(), key=str))}
    c = checkins.copy()
    c["user_id"] = c["user_raw"].map(users)
    c["poi_id"] = c["poi_raw"].map(poi_ids)
    p = pois[pois["poi_raw"].isin(poi_ids)].copy()
    p["poi_id"] = p["poi_raw"].map(poi_ids)
    p = p.sort_values("poi_id").reset_index(drop=True)
    return c, p


def dataset_stats(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    def _c(df: pd.DataFrame, user="user_raw", poi="poi_raw") -> dict:
        u = user if user in df.columns else "user_id"
        p = poi if poi in df.columns else "poi_id"
        return {
            "n_users": int(df[u].nunique()),
            "n_pois": int(df[p].nunique()),
            "n_checkins": int(len(df)),
        }

    return {"before": _c(before), "after": _c(after, user="user_id", poi="poi_id")}


def load_raw(dataset: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bbox = cfg["bbox"]
    if dataset == "synthetic":
        syn = cfg.get("synthetic", {})
        return generate_synthetic(
            n_users=int(syn.get("n_users", 80)),
            n_pois=int(syn.get("n_pois", 40)),
            checkins_per_user=int(syn.get("checkins_per_user", 24)),
            n_children=int(syn.get("n_children", 4)),
            bbox=bbox,
            seed=int(cfg.get("seed", 0)),
        )
    path = download_dataset(dataset)
    if dataset == "foursquare_nyc":
        return _parse_foursquare(path, bbox)
    if dataset == "gowalla":
        return _parse_gowalla(path, bbox)
    raise ValueError(dataset)


def preprocess_dataset(dataset: str, seed: int | None = None) -> Path:
    cfg = load_config(dataset)
    if seed is not None:
        cfg["seed"] = seed
    purpose = cfg.get("purpose", "location_analytics_research")
    pp = cfg["preprocess"]

    raw_checkins, raw_pois = load_raw(dataset, cfg)
    before = raw_checkins.copy()
    filtered = iterative_filter(raw_checkins, pp["min_user_checkins"], pp["min_poi_visits"])
    checkins, pois = reindex(filtered, raw_pois)

    crs = LocalCRS.from_bbox(**cfg["bbox"])
    checkins = add_xy(checkins, crs)
    pois = add_xy(pois, crs)
    checkins = add_sessions(checkins, pp["session_gap_hours"])
    checkins = filter_children(checkins)

    members, nonmembers = stratified_user_split(checkins, pp["member_frac"], int(cfg["seed"]))
    checkins = assign_time_splits(checkins, members, pp["train_frac"], pp["val_frac"])

    consent = default_consent_table(sorted(checkins["user_id"].unique()), purpose=purpose)
    checkins = filter_by_consent(checkins, consent, purpose)

    keep = [c for c in ALLOWED_COLUMNS if c in checkins.columns]
    checkins = checkins[keep]
    pois_out = pois[["poi_id", "lat", "lon", "x", "y"] + (["category"] if "category" in pois else [])]

    out_dir = resolve_path(cfg, "processed") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    checkins.to_parquet(out_dir / "checkins.parquet", index=False)
    pois_out.to_parquet(out_dir / "pois.parquet", index=False)
    consent.to_parquet(out_dir / "consent.parquet", index=False)

    stats = dataset_stats(before, checkins)
    stats["crs"] = crs.epsg
    stats["n_members"] = int(len(members))
    stats["n_nonmembers"] = int(len(nonmembers))
    stats["n_train"] = int((checkins["split"] == "train").sum())
    stats["n_val"] = int((checkins["split"] == "val").sum())
    stats["n_test"] = int((checkins["split"] == "test").sum())
    stats["purpose"] = purpose
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    split = {
        "members": members.tolist(),
        "nonmembers": nonmembers.tolist(),
        "seed": int(cfg["seed"]),
        "crs": crs.epsg,
        "bbox": cfg["bbox"],
        "purpose": purpose,
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return out_dir


def load_processed(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = load_config(dataset)
    d = resolve_path(cfg, "processed") / dataset
    checkins = pd.read_parquet(d / "checkins.parquet")
    pois = pd.read_parquet(d / "pois.parquet")
    split = json.loads((d / "split.json").read_text(encoding="utf-8"))
    return checkins, pois, split


def assert_minimised_schema(checkins: pd.DataFrame) -> None:
    extra = set(checkins.columns) - set(ALLOWED_COLUMNS)
    if extra:
        raise AssertionError(f"Unexpected columns (data minimisation): {extra}")
