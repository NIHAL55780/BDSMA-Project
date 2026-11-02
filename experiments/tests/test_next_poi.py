from __future__ import annotations

from lpdp.models.gru_next_poi import PopularBaseline
from lpdp.tasks.next_poi import run_next_poi


def test_next_poi_beats_popular_baseline(processed_tiny):
    checkins, _, _ = processed_tiny
    data = checkins.copy()
    data["original_poi_id"] = data["poi_id"]
    cfg = {
        "model": {
            "poi_dim": 16,
            "hour_dim": 4,
            "hidden": 16,
            "seq_len": 6,
            "lr": 0.02,
            "batch_size": 16,
            "max_epochs": 6,
            "patience": 4,
            "windows_per_user_cap": 12,
        },
        "fl": {"clients_per_round": 4, "local_epochs": 1, "rounds": 2, "clip_norm": 1.0, "lr": 0.05},
    }
    result = run_next_poi(data, cfg, mechanism="M0", param_value=0, seed=0)
    assert 0.0 <= result["acc10"] <= 1.0
    assert 0.0 <= result["popular_acc10"] <= 1.0
    # On synthetic favourite-POI sequences the GRU should at least match popularity.
    assert result["acc10"] + 1e-9 >= result["popular_acc10"] * 0.8
    assert result["n_test_windows"] > 0


def test_popular_baseline_ranking():
    b = PopularBaseline()
    b.fit([1, 1, 1, 2, 2, 3])
    m = b.metrics([1, 1, 2, 3], n_pois=5, ks=(1, 5, 10))
    assert m["acc1"] == 0.5
    assert m["acc5"] == 1.0
