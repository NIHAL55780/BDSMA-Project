"""GRU next-POI model and a most-popular-POI baseline."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch import nn


class GRUNextPOI(nn.Module):
    def __init__(
        self,
        n_pois: int,
        poi_dim: int = 128,
        hour_dim: int = 16,
        hidden: int = 256,
        n_hours: int = 168,
        use_dp_rnn: bool = False,
    ) -> None:
        super().__init__()
        self.n_pois = n_pois
        self.poi_emb = nn.Embedding(n_pois + 1, poi_dim, padding_idx=0)
        self.hour_emb = nn.Embedding(n_hours, hour_dim)
        gru_in = poi_dim + hour_dim
        if use_dp_rnn:
            try:
                from opacus.layers import DPGRU

                self.gru = DPGRU(gru_in, hidden, batch_first=True)
            except Exception:  # noqa: BLE001
                self.gru = nn.GRU(gru_in, hidden, batch_first=True)
        else:
            self.gru = nn.GRU(gru_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_pois + 1)

    def forward(self, poi_ids: torch.Tensor, hours: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # poi_ids, hours: [B, L] (right-padded); lengths: [B]
        x = torch.cat([self.poi_emb(poi_ids), self.hour_emb(hours)], dim=-1)
        out, _ = self.gru(x)
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(out.size(0), device=out.device), idx]
        return self.out(last)


class PopularBaseline:
    """Predict the globally most popular training POIs (utility floor)."""

    def __init__(self) -> None:
        self.ranking: list[int] = []

    def fit(self, targets: list[int]) -> None:
        counts = Counter(targets)
        self.ranking = [poi for poi, _ in counts.most_common()]

    def metrics(self, targets: list[int], n_pois: int, ks=(1, 5, 10)) -> dict[str, float]:
        ranks = []
        rank_of = {p: i for i, p in enumerate(self.ranking)}
        for t in targets:
            ranks.append(rank_of.get(t, n_pois) + 1)
        ranks_a = np.asarray(ranks, dtype=float)
        out = {f"acc{k}": float((ranks_a <= k).mean()) for k in ks}
        out["mrr"] = float((1.0 / ranks_a).mean())
        return out
