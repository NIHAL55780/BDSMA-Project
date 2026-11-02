from __future__ import annotations

from lpdp.tasks.hotspot import compare_hotspots, count_grid, make_grid, run_hotspot


def test_hotspot_f1_raw_vs_raw(processed_tiny, rng):
    checkins, _, _ = processed_tiny
    cfg = {"hotspot": {"cell_m": 400, "top_pct": 0.15, "user_cap": 3}}
    metrics = run_hotspot(checkins, checkins, cfg, mechanism="M0", param_value=0, rng=rng)
    assert metrics["hotspot_f1"] == 1.0


def test_compare_identical_counts(processed_tiny):
    checkins, _, _ = processed_tiny
    grid = make_grid(checkins, 400)
    counts = count_grid(checkins, grid)
    out = compare_hotspots(counts, counts, top_pct=0.15)
    assert out["hotspot_f1"] == 1.0
    assert out["count_mae"] == 0.0
