from __future__ import annotations

import pandas as pd
import torch

from lpdp.eval.metrics import hotspot_f1, ranking_metrics
from lpdp.eval.pareto import non_dominated


def test_ranking_metrics_perfect():
    logits = torch.tensor([[0.0, 10.0, 1.0, 0.5], [0.0, 0.1, 8.0, 0.2]])
    targets = torch.tensor([1, 2])
    m = ranking_metrics(logits, targets)
    assert m["acc1"] == 1.0
    assert m["mrr"] == 1.0


def test_hotspot_f1_sets():
    assert hotspot_f1({(0, 0), (1, 1)}, {(0, 0), (1, 1)}) == 1.0
    assert hotspot_f1(set(), set()) == 1.0
    assert hotspot_f1({(0, 0)}, {(1, 1)}) == 0.0


def test_pareto_front():
    df = pd.DataFrame(
        {
            "acc10": [0.2, 0.5, 0.4, 0.5],
            "mia_auc": [0.9, 0.6, 0.55, 0.8],
            "mechanism": list("ABCD"),
        }
    )
    front = non_dominated(df, "acc10", "mia_auc")
    mechs = set(front["mechanism"])
    assert "C" in mechs  # 0.4 util, 0.55 risk
    assert "B" in mechs  # 0.5 util, 0.6 risk
    assert "D" not in mechs  # same util as B, worse risk
    assert "A" not in mechs
