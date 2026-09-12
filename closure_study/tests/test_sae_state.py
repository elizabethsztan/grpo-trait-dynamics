import numpy as np
import pytest

from closure_study.diagnose_consistency import diagnose
from closure_study.fit_controlled import rollout
from closure_study.fit_llm import SAE_STATE, fit_series
from closure_study.refine_shape import NEXT_FIELDS, failure_audit


def state_rows():
    rows, mu, v = [], .3, .25
    for t in range(20):
        beta, gamma = .08 + .05 * mu + .12 * v, .25 - .08 * mu + .02 * mu**2
        c, m3 = beta * v, gamma * v**1.5
        q = .8 * beta * m3
        rows.append(dict(family="llm_continuous", run_id="run", trait_id="25273", setting_id="sae",
                         inspection_status="development", step=t, step_end=t+1, interval=1,
                         mu=mu, V=v, beta=beta, skewness=gamma, C=c, M3=m3, Q=q))
        mu, v = mu + c, v + q - c*c
    return rows, (mu, v)


def test_three_candidates_recover_coupled_law_and_share_shape_and_kappa():
    rows, final = state_rows()
    parameters, metrics, paths, residuals = fit_series(rows, sae_state=True)
    assert [p["model"] for p in parameters] == [name for name, _, _ in SAE_STATE]
    for p in parameters:
        np.testing.assert_allclose([p[k] for k in ("gamma0", "gamma1", "gamma2", "kappa")],
                                   [.25, -.08, .02, .8], atol=1e-10)
        assert p["selection_objective"] == "C" and p["shape_objective"] == "M3"
    p = parameters[1]
    np.testing.assert_allclose([p[k] for k in ("c0", "c1", "c2")], [.08, .05, .12], atol=1e-9)
    assert metrics[1]["status"] == "complete" and metrics[1]["mu_rmse"] < 1e-12
    last = next(r for r in paths if r["model"] == p["model"] and r["step"] == 20)
    np.testing.assert_allclose([last["mu_generated"], last["V_generated"]], final, atol=1e-12)
    for r in residuals:
        assert sum(r[f"Q_{part}_residual"] for part in ("flux", "moment", "selection")) == pytest.approx(r["Q_total_residual"], abs=1e-14)
    points, events, _ = diagnose(parameters, metrics, paths, residuals)
    assert events[1]["selection_roots"] is None and events[2]["selection_roots"] is None
    for r in points:
        p = next(p for p in parameters if p["model"] == r["model"])
        mu, v, t = r["mu_generated"], r["V_generated"], r["step"]
        beta = p["c0"] + p["c1"] * (t if p["predictor"] == "step" else mu)
        beta += p["c2"] * (v if p["predictor"] == "mu_V" else (t if p["predictor"] == "step" else mu)**2)
        assert r["beta_generated"] == pytest.approx(beta)
        if t < 20:
            assert r["next_mu"] == pytest.approx(mu + beta*v)


def test_common_mask_flux_objective_and_unchanged_mean_only_reference():
    rows, _ = state_rows()
    rows[4].update(beta=None, skewness=None)
    rows[10]["C"] *= 1.1
    rows[10]["beta"] *= 1.1
    new = fit_series(rows, sae_state=True)
    old = fit_series(rows, sae_shape=True)
    for a, b in zip(new, old):
        a = [r for r in a if r["model"] == "flux_state_quadratic"]
        b = [r for r in b if r["model"] == "moment_state_quadratic"]
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.keys() == y.keys()
            for k in x.keys() - {"model"}:
                if isinstance(x[k], (float, np.floating)):
                    np.testing.assert_allclose(x[k], y[k], rtol=0., atol=0., equal_nan=True)
                else:
                    assert x[k] == y[k]
    assert all(p["fit_points"] == 19 for p in new[0])
    assert all(r["step"] != 4 for r in new[3])
    for p in new[0]:
        rs = [r for r in new[3] if r["model"] == p["model"]]
        mu, v, t, error = (np.array([r[k] for r in rs]) for k in ("mu", "V", "step", "C_residual"))
        basis = np.column_stack([np.ones_like(mu), t if p["predictor"] == "step" else mu,
                                 v if p["predictor"] == "mu_V" else (t if p["predictor"] == "step" else mu)**2])
        np.testing.assert_allclose((v[:, None]*basis).T @ error, 0., atol=1e-12)
        assert sum(error**2) == pytest.approx(sum(r["V"]**2 * r["beta_residual"]**2 for r in rs))


def test_variance_selection_stops_invalid_mean_without_clipping():
    path, status = rollout([.1, 1.], [0, 1], [-.2, 0., -.1], [0.],
                           selection_variance=True, mean_bounds=(0., np.inf))
    assert status == "mean_outside_support_at_1"
    assert np.isnan(path[1:]).all()
    base = dict(run_id="run", trait_id="25273", model="flux_state_variance")
    p = dict(base, c0=-.2, c1=0., c2=-.1, predictor="mu_V", gamma_predictor="mu",
             gamma0=0., gamma1=0., gamma2=0., kappa=1.)
    paths = [dict(base, step=t, mu_generated=x[0], V_generated=x[1]) for t, x in enumerate(path)]
    residual = dict(base, step=0, mu=.1, V=1., **{k: .91 if k.startswith("V_") else -.2 for k in NEXT_FIELDS})
    audit = failure_audit([p], [dict(base, status=status)], paths, [residual])[0]
    assert audit["attempted_next_mu"] == pytest.approx(-.2)
    assert audit["attempted_next_V"] == pytest.approx(.91)
