import numpy as np

from src.price import CumulativePriceTracker, price_stats


def test_self_normalized_price_is_zero_when_omega_is_one():
    stats = price_stats(np.ones(3), np.array([0.0, 1.0, 1.0]))

    assert stats["raw_cov_step"] == 0.0
    assert stats["sn_step"] == 0.0


def test_self_normalized_price_differs_from_raw_when_mean_omega_not_one():
    omega = np.array([2.0, 1.0, 1.0])
    trait = np.array([1.0, 0.0, 0.0])

    stats = price_stats(omega, trait)

    assert np.isclose(stats["raw_cov_step"], np.mean(omega * trait) - np.mean(omega) * np.mean(trait))
    assert np.isclose(stats["sn_step"], np.sum(omega * trait) / np.sum(omega) - np.mean(trait))
    assert not np.isclose(stats["raw_cov_step"], stats["sn_step"])


def test_cumulative_tracker_stores_raw_and_self_normalized_totals():
    tracker = CumulativePriceTracker()

    first = tracker.update_price("eval", "output_agreement", raw_step=0.2, sn_step=0.3)
    second = tracker.update_price("eval", "output_agreement", raw_step=-0.1, sn_step=0.4)

    assert first == {"raw_cov_cum": 0.2, "sn_cum": 0.3}
    assert second == {"raw_cov_cum": 0.1, "sn_cum": 0.7}
