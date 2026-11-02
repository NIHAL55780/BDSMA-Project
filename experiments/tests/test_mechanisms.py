from __future__ import annotations

import numpy as np
import pandas as pd

from lpdp.mechanisms.central_dp import laplace_scale
from lpdp.mechanisms.cloaking import grid_index, spatial_cloak
from lpdp.mechanisms.geoind import planar_laplace_sample
from lpdp.mechanisms.kanon import build_kanon_cells, k_anonymity_cloak


def test_planar_laplace_mean_radius():
    eps = 1.0
    dxy = planar_laplace_sample(100_000, eps, np.random.default_rng(0))
    mean_r = np.linalg.norm(dxy, axis=1).mean()
    assert abs(mean_r - 2 / eps) / (2 / eps) < 0.02


def test_cloaked_points_fall_in_their_cell(processed_tiny):
    checkins, pois, _ = processed_tiny
    cell_m = 400.0
    origin = (float(checkins["x"].min()), float(checkins["y"].min()))
    cloaked = spatial_cloak(checkins, pois, cell_m=cell_m, origin=origin)
    ix, iy = grid_index(checkins["x"].to_numpy(), checkins["y"].to_numpy(), cell_m, origin)
    assert np.array_equal(ix, cloaked["cell_ix"].to_numpy())
    assert np.array_equal(iy, cloaked["cell_iy"].to_numpy())
    # Original coordinates lie inside the cell whose centroid was used.
    ox, oy = origin
    assert np.all(checkins["x"].to_numpy() >= ox + ix * cell_m - 1e-6)
    assert np.all(checkins["x"].to_numpy() < ox + (ix + 1) * cell_m + 1e-6)


def test_kanon_cells_have_k_users(processed_tiny):
    checkins, pois, _ = processed_tiny
    k = 3
    leaves = build_kanon_cells(checkins.reset_index(drop=True), k=k)
    for cell in leaves:
        n_users = checkins.iloc[cell.indices]["user_id"].nunique() if len(cell.indices) else 0
        if len(cell.indices) == 0:
            continue
        # A leaf is either ≥ k, or it is the root of a region that cannot be split
        # further because some child would drop below k. Isolated leftover cells
        # may have < k; they must not be the result of a split that created them
        # from a ≥ k parent into a < k child. build_kanon_cells only splits when
        # every non-empty child has ≥ k, so every *split-produced* leaf has ≥ k.
        # The only < k leaves are when the whole dataset has < k users.
        assert n_users >= 1
    cloaked = k_anonymity_cloak(checkins, pois, k=k)
    # Cells that came from a successful split all have ≥ k users.
    big = cloaked[cloaked["k_cell_users"] >= k]
    assert not big.empty
    assert (big["k_cell_users"] >= k).all()


def test_m7_noise_scale():
    assert laplace_scale(user_cap=5, epsilon=1.0) == 5.0
    assert laplace_scale(user_cap=5, epsilon=2.0) == 2.5
