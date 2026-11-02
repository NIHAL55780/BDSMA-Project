"""Consent, purpose limitation, child filter, and erasure helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_PURPOSE = "location_analytics_research"


def default_consent_table(user_ids: list[int], purpose: str = DEFAULT_PURPOSE) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "purpose": purpose,
            "granted_at": now,
            "withdrawn_at": pd.NA,
        }
    )


def consenting_users(consent: pd.DataFrame, purpose: str) -> set[int]:
    active = consent[
        (consent["purpose"] == purpose)
        & (consent["withdrawn_at"].isna() | (consent["withdrawn_at"] == ""))
    ]
    return set(active["user_id"].astype(int).tolist())


def filter_by_consent(
    checkins: pd.DataFrame,
    consent: pd.DataFrame,
    purpose: str,
) -> pd.DataFrame:
    allowed = consenting_users(consent, purpose)
    return checkins[checkins["user_id"].isin(allowed)].copy()


def filter_children(checkins: pd.DataFrame) -> pd.DataFrame:
    if "is_child" not in checkins.columns:
        return checkins.copy()
    return checkins[checkins["is_child"].fillna(False).astype(bool) == False].copy()  # noqa: E712


def erase_user(processed_dir: Path, user_id: int) -> None:
    """Remove a user from every artefact under a processed dataset directory."""
    checkins_path = processed_dir / "checkins.parquet"
    consent_path = processed_dir / "consent.parquet"
    if checkins_path.exists():
        df = pd.read_parquet(checkins_path)
        df = df[df["user_id"] != user_id]
        df.to_parquet(checkins_path, index=False)
    if consent_path.exists():
        c = pd.read_parquet(consent_path)
        now = datetime.now(timezone.utc).isoformat()
        mask = c["user_id"] == user_id
        c.loc[mask, "withdrawn_at"] = now
        c.to_parquet(consent_path, index=False)
    released = processed_dir / "released"
    if released.exists():
        for path in released.glob("*.parquet"):
            df = pd.read_parquet(path)
            if "user_id" in df.columns:
                df[df["user_id"] != user_id].to_parquet(path, index=False)
