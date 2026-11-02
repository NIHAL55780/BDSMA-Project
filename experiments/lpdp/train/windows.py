"""Session windows for next-POI prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Window:
    user_id: int
    poi_ids: list[int]
    hours: list[int]
    target: int
    split: str


def hour_of_week(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    return int(t.dayofweek) * 24 + int(t.hour)


def build_windows(checkins: pd.DataFrame, seq_len: int) -> list[Window]:
    """History uses (possibly privatised) poi_id; target uses original_poi_id."""
    df = checkins.sort_values(["user_id", "session_id", "ts"])
    if "original_poi_id" not in df.columns:
        df = df.copy()
        df["original_poi_id"] = df["poi_id"]
    windows: list[Window] = []
    for (uid, _sid), g in df.groupby(["user_id", "session_id"], sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        pois = g["poi_id"].astype(int).tolist()
        targets = g["original_poi_id"].astype(int).tolist()
        hours = [hour_of_week(t) for t in g["ts"]]
        split = str(g["split"].iloc[-1]) if "split" in g.columns else "train"
        splits = g["split"].tolist() if "split" in g.columns else ["train"] * len(g)
        for i in range(1, len(g)):
            start = max(0, i - seq_len)
            windows.append(
                Window(
                    user_id=int(uid),
                    poi_ids=pois[start:i],
                    hours=hours[start:i],
                    target=int(targets[i]),
                    split=str(splits[i]),
                )
            )
    return windows


def cap_windows_per_user(windows: list[Window], cap: int, rng: np.random.Generator) -> list[Window]:
    by_user: dict[int, list[Window]] = {}
    for w in windows:
        by_user.setdefault(w.user_id, []).append(w)
    out: list[Window] = []
    for uid, ws in by_user.items():
        if len(ws) <= cap:
            out.extend(ws)
        else:
            idx = rng.choice(len(ws), size=cap, replace=False)
            out.extend(ws[i] for i in idx)
    return out


class WindowDataset(Dataset):
    def __init__(self, windows: list[Window], seq_len: int) -> None:
        self.windows = windows
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Window:
        return self.windows[i]


def collate_windows(batch: list[Window]) -> dict[str, torch.Tensor]:
    seq_len = max(len(w.poi_ids) for w in batch)
    b = len(batch)
    poi = torch.zeros(b, seq_len, dtype=torch.long)
    hours = torch.zeros(b, seq_len, dtype=torch.long)
    lengths = torch.zeros(b, dtype=torch.long)
    targets = torch.zeros(b, dtype=torch.long)
    users = torch.zeros(b, dtype=torch.long)
    for i, w in enumerate(batch):
        n = len(w.poi_ids)
        poi[i, :n] = torch.tensor(w.poi_ids, dtype=torch.long)
        hours[i, :n] = torch.tensor(w.hours, dtype=torch.long)
        lengths[i] = n
        targets[i] = w.target
        users[i] = w.user_id
    return {
        "poi_ids": poi,
        "hours": hours,
        "lengths": lengths,
        "targets": targets,
        "users": users,
    }
