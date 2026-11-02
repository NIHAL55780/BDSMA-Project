"""(ε, δ)-DP accounting for subsampled Gaussian (GDP conversion).

When Opacus is installed, M6 uses its RDPAccountant. This module is the
fallback and also used to invert a target ε into a noise multiplier.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _mu_gdp(noise_multiplier: float, steps: int, sample_rate: float) -> float:
    if noise_multiplier <= 0:
        return 1e6
    return float(sample_rate * math.sqrt(steps) / noise_multiplier)


def _delta_from_eps(eps: float, mu: float) -> float:
    if mu <= 0:
        return 0.0
    return float(
        norm.cdf(-eps / mu + mu / 2.0) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2.0)
    )


def gaussian_dp_epsilon(
    noise_multiplier: float,
    steps: int,
    sample_rate: float,
    delta: float,
    lo: float = 0.0,
    hi: float = 50.0,
) -> float:
    """Binary-search ε such that μ-GDP implies (ε, δ)-DP."""
    mu = _mu_gdp(noise_multiplier, steps, sample_rate)
    if mu <= 0:
        return 0.0
    # If even huge eps cannot meet delta, return hi.
    for _ in range(60):
        mid = (lo + hi) / 2
        if _delta_from_eps(mid, mu) > delta:
            lo = mid
        else:
            hi = mid
    return float(hi)


def noise_for_target_eps(
    target_eps: float,
    delta: float,
    steps: int,
    sample_rate: float,
) -> float:
    lo, hi = 0.1, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        eps = gaussian_dp_epsilon(mid, steps, sample_rate, delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    return float(hi)


def opacus_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float | None:
    try:
        from opacus.accountants import RDPAccountant

        acc = RDPAccountant()
        for _ in range(steps):
            acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return float(acc.get_epsilon(delta=delta))
    except Exception:  # noqa: BLE001
        return None


def report_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    eps = opacus_epsilon(noise_multiplier, sample_rate, steps, delta)
    if eps is not None:
        return eps
    return gaussian_dp_epsilon(noise_multiplier, steps, sample_rate, delta)
