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
    totals: dict[str, dict[str, object]]

    def __init__(self):
        self.totals = {}

    def update(self, distribution: str, trait_name: str, cov_step: float) -> float:
        self.totals.setdefault(distribution, {})
        current = self.totals[distribution].get(trait_name, 0.0)
        if isinstance(current, dict):
            current = current.get("raw_cov_cum", 0.0)
        self.totals[distribution][trait_name] = float(current) + cov_step
        return float(self.totals[distribution][trait_name])

    def update_price(self, distribution: str, trait_name: str, raw_step: float, sn_step: float) -> dict:
        self.totals.setdefault(distribution, {})
        current = self.totals[distribution].get(trait_name, {"raw_cov_cum": 0.0, "sn_cum": 0.0})
        if not isinstance(current, dict):
            current = {"raw_cov_cum": float(current), "sn_cum": 0.0}
        updated = {
            "raw_cov_cum": float(current.get("raw_cov_cum", 0.0)) + float(raw_step),
            "sn_cum": float(current.get("sn_cum", 0.0)) + float(sn_step),
        }
        self.totals[distribution][trait_name] = updated
        return updated


def price_stats(omega, trait, cumulative: float = 0.0) -> dict:
    omega = np.asarray(omega, dtype=float)
    trait = np.asarray(trait, dtype=float)
    raw_cov = price_covariance(omega, trait)
    mean_omega = float(np.mean(omega)) if omega.size else 0.0
    pre_trait_mean = float(np.mean(trait)) if trait.size else 0.0
    if omega.size and np.sum(omega) > 0:
        post_trait_mean_importance_weighted = float(np.sum(omega * trait) / np.sum(omega))
        sn_step = float(post_trait_mean_importance_weighted - pre_trait_mean)
    else:
        post_trait_mean_importance_weighted = 0.0
        sn_step = 0.0
    raw_cov_cum = cumulative + raw_cov
    return {
        "raw_cov_step": raw_cov,
        "raw_cov_cum": raw_cov_cum,
        "sn_step": sn_step,
        "sn_cum": cumulative + sn_step,
        "cov_step": raw_cov,
        "cov_cum": raw_cov_cum,
        "mean_omega": mean_omega,
        "std_omega": float(np.std(omega)) if omega.size else 0.0,
        "min_omega": float(np.min(omega)) if omega.size else 0.0,
        "max_omega": float(np.max(omega)) if omega.size else 0.0,
        "ess": effective_sample_size(omega),
        "n": int(omega.size),
        "pre_trait_mean": pre_trait_mean,
        "post_trait_mean_importance_weighted": post_trait_mean_importance_weighted,
    }
