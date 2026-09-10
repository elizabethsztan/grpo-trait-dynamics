import numpy as np
import pytest

from closure_study.statistics import (conditional_bins, enumerated_transition, flux_statistics,
                                     moments, selection_fields)


def test_binary_reduction_and_discrete_variance_correction():
    z, p, q = np.array([0., 1.]), np.array([.6, .4]), np.array([.3, .7])
    stats = enumerated_transition(z, p, q)
    selection = selection_fields(stats["mu"], stats["V"], stats["C"], binary=True)
    np.testing.assert_allclose(stats["C"], .3)
    np.testing.assert_allclose(selection["S"], 1.25)
    np.testing.assert_allclose(stats["Q"], .06)
    np.testing.assert_allclose(stats["variance_change_from_flux"], -.03)
    assert abs(stats["variance_identity_residual"]) < 1e-15


def test_linear_conditional_multiplier_implies_higher_moment_flux():
    z, p, beta = np.array([-1., 0., 2.]), np.array([.2, .5, .3]), .12
    mu = p @ z
    w = 1 + beta * (z - mu)
    stats = flux_statistics(z, w, p)
    np.testing.assert_allclose(stats["beta"], beta)
    assert abs(stats["Q_minus_beta_M3"]) < 1e-15
    exact = enumerated_transition(z, p, p * w)
    np.testing.assert_allclose(stats["variance_change_from_flux"], exact["V_next"] - exact["V"])


def test_mean_identity_does_not_establish_conditional_linearity():
    z, p = np.array([0., 1., 3.]), np.array([.25, .5, .25])
    w = np.exp(.3 * z * z)
    w /= p @ w
    stats = flux_statistics(z, w, p)
    np.testing.assert_allclose(stats["C"], stats["beta"] * stats["V"])
    assert abs(stats["Q_minus_beta_M3"]) > .01
    exact = enumerated_transition(z, p, p * w)
    assert abs(exact["variance_identity_residual"]) < 1e-14


def test_translation_scaling_permutation_and_population_mass_invariance():
    z, w, p = np.array([-2., 0., 1., 4.]), np.array([.7, .8, 1., 1.9]), np.array([.1, .2, .3, .4])
    original = flux_statistics(z, w, p)
    changed = flux_statistics((3 * z + 100)[::-1], w[::-1], (7 * p)[::-1])
    for name, factor in [("C", 3), ("V", 9), ("M3", 27), ("beta", 1/3), ("Q", 9), ("Q_minus_beta_M3", 9)]:
        np.testing.assert_allclose(changed[name], factor * original[name], rtol=1e-12, atol=1e-12)


def test_raw_ratios_are_not_self_normalized_silently():
    z, w = np.array([0., 1., 2.]), np.array([1., 2., 4.])
    out = flux_statistics(z, w)
    assert out["mean_omega"] != 1
    assert out["C"] != out["C_self_normalized"]
    np.testing.assert_allclose(out["C_known_normalizer"] - out["C"], out["mu"] * (out["mean_omega"] - 1))


def test_zero_variance_and_noop():
    out = flux_statistics([4., 4.], [1., 1.])
    assert out["beta"] is None and out["Q_minus_beta_M3"] is None
    assert out["C"] == out["Q"] == 0
    identical = enumerated_transition([-1., 3.], [.25, .75], [.25, .75])
    assert identical["C"] == identical["Q"] == 0


def test_support_failure_is_explicit():
    assert enumerated_transition([0., 1.], [1., 0.], [.5, .5])["support_failure"]


def test_conditional_bins_preserve_weighted_population_and_raw_normalizer():
    z, p = np.array([0., 0., 1., 2., 5.]), np.array([.1, .1, .3, .1, .4])
    w = 1.2 + .1 * (z - p @ z)
    bins = conditional_bins(z, w, p, count=3)
    np.testing.assert_allclose(sum(r["population_mass"] for r in bins), 1)
    np.testing.assert_allclose(sum(r["population_mass"] * r["omega_mean"] for r in bins), p @ w)
    np.testing.assert_allclose([r["conditional_residual"] for r in bins], .2)


@pytest.mark.parametrize("scores,omega", [([0., np.nan], [1., 1.]), ([0., 1.], [1., -1.]), ([0., 1.], [1., np.inf])])
def test_invalid_inputs_rejected(scores, omega):
    with pytest.raises(ValueError):
        flux_statistics(scores, omega)


def test_overflow_rejected():
    with pytest.raises(FloatingPointError):
        moments([-1e100, 1e100])
