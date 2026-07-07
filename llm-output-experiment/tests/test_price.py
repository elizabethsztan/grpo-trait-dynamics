import numpy as np

from src.price import price_covariance


def test_price_covariance_matches_definition():
    omega = np.array([0.8, 1.2, 1.5, 0.5])
    trait = np.array([0.0, 1.0, 1.0, 0.0])

    cov = price_covariance(omega, trait)

    assert np.isclose(cov, np.mean(omega * trait) - np.mean(omega) * np.mean(trait))


def test_finite_distribution_price_identity():
    p_t = np.array([0.2, 0.3, 0.5])
    p_tp1 = np.array([0.1, 0.4, 0.5])
    trait = np.array([-1.0, 0.5, 2.0])

    omega = p_tp1 / p_t
    drift = np.sum(p_tp1 * trait) - np.sum(p_t * trait)
    cov = price_covariance(omega, trait, weights=p_t)

    assert np.isclose(drift, cov)
