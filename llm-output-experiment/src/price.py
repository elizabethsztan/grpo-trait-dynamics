from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def price_covariance(omega, trait, weights=None) -> float:
    omega = np.asarray(omega, dtype=float)
    trait = np.asarray(trait, dtype=float)
    if omega.shape != trait.shape:
        raise ValueError("omega and trait must have the same shape")

    if weights is None:
        return float(np.mean(omega * trait) - np.mean(omega) * np.mean(trait))

    weights = np.asarray(weights, dtype=float)
    if weights.shape != omega.shape:
        raise ValueError("weights must have the same shape as omega")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / total
    return float(np.sum(weights * omega * trait) - np.sum(weights * omega) * np.sum(weights * trait))


def effective_sample_size(omega) -> float:
    omega = np.asarray(omega, dtype=float)
    denom = np.sum(omega**2)
    if denom <= 0:
        return 0.0
    return float(np.sum(omega) ** 2 / denom)


@dataclass
class CumulativePriceTracker:
    totals: dict[str, dict[str, float]]

    def __init__(self):
        self.totals = {}

    def update(self, distribution: str, trait_name: str, cov_step: float) -> float:
        self.totals.setdefault(distribution, {})
        self.totals[distribution][trait_name] = self.totals[distribution].get(trait_name, 0.0) + cov_step
        return self.totals[distribution][trait_name]


def price_stats(omega, trait, cumulative: float = 0.0) -> dict:
    omega = np.asarray(omega, dtype=float)
    trait = np.asarray(trait, dtype=float)
    cov = price_covariance(omega, trait)
    return {
        "cov_step": cov,
        "cov_cum": cumulative + cov,
        "mean_omega": float(np.mean(omega)),
        "std_omega": float(np.std(omega)),
        "min_omega": float(np.min(omega)),
        "max_omega": float(np.max(omega)),
        "ess": effective_sample_size(omega),
        "n": int(omega.size),
    }
