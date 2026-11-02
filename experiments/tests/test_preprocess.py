from __future__ import annotations

import pandas as pd

from lpdp.data.consent import (
    default_consent_table,
    erase_user,
    filter_by_consent,
    filter_children,
)
from lpdp.data.preprocess import ALLOWED_COLUMNS, assert_minimised_schema, iterative_filter, preprocess_dataset
from lpdp.data.splits import assign_time_splits, stratified_user_split


def test_iterative_filter_thresholds(synthetic_raw):
    checkins, _ = synthetic_raw
    out = iterative_filter(checkins, min_user=8, min_poi=5)
    assert (out.groupby("user_raw").size() >= 8).all()
    assert (out.groupby("poi_raw").size() >= 5).all()


def test_child_filter(synthetic_raw):
    checkins, _ = synthetic_raw
    assert checkins["is_child"].any()
    clean = filter_children(checkins)
    assert not clean["is_child"].any()
    assert set(clean["user_raw"]) <= set(checkins.loc[~checkins["is_child"], "user_raw"])


def test_consent_purpose_filter(processed_tiny):
    checkins, _, _ = processed_tiny
    users = sorted(checkins["user_id"].unique())
    consent = default_consent_table(users, purpose="location_analytics_research")
    consent.loc[consent["user_id"] == users[0], "purpose"] = "other"
    kept = filter_by_consent(checkins, consent, "location_analytics_research")
    assert users[0] not in set(kept["user_id"])
    assert filter_by_consent(checkins, consent, "missing_purpose").empty


def test_erasure_removes_user(tmp_path, processed_tiny):
    checkins, _, _ = processed_tiny
    uid = int(checkins["user_id"].iloc[0])
    checkins.to_parquet(tmp_path / "checkins.parquet", index=False)
    default_consent_table(sorted(checkins["user_id"].unique())).to_parquet(
        tmp_path / "consent.parquet", index=False
    )
    erase_user(tmp_path, uid)
    after = pd.read_parquet(tmp_path / "checkins.parquet")
    assert uid not in set(after["user_id"])


def test_time_and_user_splits(processed_tiny):
    checkins, _, _ = processed_tiny
    members, nonmembers = stratified_user_split(checkins, member_frac=0.7, seed=0)
    assert set(members).isdisjoint(set(nonmembers))
    assert abs(len(members) / (len(members) + len(nonmembers)) - 0.7) < 0.15
    split = assign_time_splits(checkins, members, 0.8, 0.1)
    mem = split[split["user_id"].isin(members)]
    assert {"train", "val", "test"} <= set(mem["split"])
    hold = split[split["user_id"].isin(nonmembers)]
    assert set(hold["split"]) == {"holdout"}


def test_preprocess_synthetic_writes_stats():
    out = preprocess_dataset("synthetic", seed=0)
    stats = (out / "dataset_stats.json").read_text(encoding="utf-8")
    assert "n_users" in stats
    checkins = pd.read_parquet(out / "checkins.parquet")
    extra = set(checkins.columns) - set(ALLOWED_COLUMNS)
    assert not extra
    assert_minimised_schema(checkins)
