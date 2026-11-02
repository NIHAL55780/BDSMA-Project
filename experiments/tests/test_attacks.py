from __future__ import annotations

from lpdp.attacks.mia import membership_inference
from lpdp.attacks.reid import reidentification
from lpdp.models.gru_next_poi import GRUNextPOI
from lpdp.train.windows import Window, build_windows


def test_reid_raw_beats_chance(processed_tiny, rng):
    checkins, _, _ = processed_tiny
    out = reidentification(
        checkins,
        checkins,
        p_list=[2, 4],
        n_targets=20,
        d_m=250,
        dt_hours=1,
        rng=rng,
    )
    n_users = checkins["user_id"].nunique()
    chance = 1.0 / max(n_users, 1)
    assert out["reid_top1_p4"] > chance * 5
    assert out["reid_top1_p2"] >= out["reid_top1_p4"] * 0.5


def test_mia_untrained_near_half(processed_tiny, rng):
    checkins, _, _ = processed_tiny
    checkins = checkins.copy()
    checkins["original_poi_id"] = checkins["poi_id"]
    windows = build_windows(checkins, seq_len=6)
    if len(windows) < 10:
        return
    members = windows[: len(windows) // 2]
    non = windows[len(windows) // 2 :]
    # Re-tag user ids so the two pools are disjoint.
    for w in non:
        w.user_id += 10_000
    n_pois = int(checkins["poi_id"].max())
    model = GRUNextPOI(n_pois=n_pois, poi_dim=8, hour_dim=4, hidden=8)
    model.eval()
    mia = membership_inference(model, members, non, rng=rng, max_users=20)
    assert 0.25 <= mia["mia_auc"] <= 0.75
