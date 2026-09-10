import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from closure_study.fit_controlled import rollout
from closure_study.fit_llm import fit_series
from closure_study.refine_shape import failure_audit
from closure_study.tests.test_fit_llm import sae_rows


def shape_rows():
    rows, mu, v = [], .5, .2
    for step in range(12):
        beta, gamma = .1 + .01 * mu + .001 * mu**2, .2 + .02 * step - .001 * step**2
        c, m3 = beta * v, gamma * v**1.5
        q = .8 * beta * m3
        rows.append(dict(family="llm_continuous", run_id="sae-rerun-s2", trait_id="25273", setting_id="sae",
                         inspection_status="development", step=step, step_end=step+1, interval=1,
                         mu=mu, V=v, beta=beta, C=c, M3=m3, skewness=gamma, Q=q,
                         direct_mu_old=mu, direct_mu_new=mu+c))
        mu, v = mu + c, v + q - c*c
    return rows, (mu, v)


def test_time_shape_uses_clock_while_selection_uses_generated_mean():
    rows, final = shape_rows()
    parameters, metrics, paths, residuals = fit_series(rows, sae_shape=True)
    for p in parameters:
        np.testing.assert_allclose([p["c0"], p["c1"], p["c2"], p["kappa"]], [.1, .01, .001, .8], atol=1e-11)
        assert (p["gamma2"] is not None) == p["model"].endswith("quadratic")
    p = next(r for r in parameters if r["model"] == "moment_time_quadratic")
    np.testing.assert_allclose([p["gamma0"], p["gamma1"], p["gamma2"]], [.2, .02, -.001], atol=1e-12)
    m = next(r for r in metrics if r["model"] == p["model"])
    assert m["status"] == "complete" and m["mu_rmse"] < 1e-12 and m["V_rmse"] < 1e-12
    last = next(r for r in paths if r["model"] == p["model"] and r["step"] == 12)
    np.testing.assert_allclose([last["mu_generated"], last["V_generated"]], final, atol=1e-12)
    for r in residuals:
        assert r["Q_flux_residual"] + r["Q_moment_residual"] + r["Q_selection_residual"] == pytest.approx(r["Q_total_residual"], abs=1e-14)
        if r["model"] == p["model"]:
            assert r["delta_V_residual"] == pytest.approx(0., abs=1e-14)


def test_moment_objective_and_reference_keep_common_mask_and_observations():
    rows = sae_rows()
    rows[5].update(V=0., C=0., Q=0., M3=0., beta=None, skewness=None)
    prior = fit_series(rows, sae_selection=True)
    parameters, metrics, paths, residuals = fit_series(rows, sae_shape=True)
    old = next(r for r in prior[0] if r["model"] == "flux_state_quadratic")
    assert all([p[k] for k in ("c0", "c1", "c2", "kappa")] == [old[k] for k in ("c0", "c1", "c2", "kappa")] for p in parameters)
    assert all(p["fit_points"] == 11 for p in parameters)
    assert all(r["step"] != 5 for r in residuals)
    assert all(r["V_observed"] == 0 for r in paths if r["step"] == 5)
    reference = next(r for r in metrics if r["model"] == "gamma_ols_affine")
    old_metric = next(r for r in prior[1] if r["model"] == "flux_state_quadratic")
    assert all(reference[k] == v for k, v in old_metric.items() if k != "model")
    for model in {r["model"] for r in residuals}:
        rs = [r for r in residuals if r["model"] == model]
        if model == "gamma_ols_affine":
            assert all(r["gamma_fit_weight"] == 1 for r in rs)
        else:
            assert sum(r["M3_residual"]**2 for r in rs) == pytest.approx(sum(r["gamma_fit_weight"] * r["gamma_residual"]**2 for r in rs))
            assert all(r["gamma_fit_weight"] == pytest.approx(r["V"]**3) for r in rs)


@pytest.mark.parametrize("mu,v,b,g,m3,reason,observed_bad,outside", [
    (1., 1., .2, -10., 0., "negative_variance", 1, False),
    (.2, .5, -.2, -10., 1.15, "mean_outside_support", 0, True)])
def test_failure_audit_separates_observed_states_from_generated_failure(mu, v, b, g, m3, reason, observed_bad, outside):
    base = dict(run_id="run", trait_id="25273", model="gamma_ols_affine")
    c, q = b*v, b*m3
    path, status = rollout([mu, v], [0, 1, 2], [b], [g], mean_bounds=(0., np.inf))
    parameters = [{**base, "c0": b, "c1": 0., "c2": 0., "gamma0": g, "gamma1": 0., "gamma2": None,
                   "kappa": 1., "gamma_predictor": "mu"}]
    paths = [{**base, "step": i, "mu_generated": x[0], "V_generated": x[1]} for i, x in enumerate(path)]
    fitted_v = v + b*g*v**1.5 - c*c
    residuals = [{**base, "step": 0, "mu": mu, "V": v, "V_next_fitted": fitted_v,
                  "V_next_measured_M3": v+q-c*c, "V_next_measured_beta": fitted_v,
                  "V_next_measured_flux": v+q-c*c, "mu_next_fitted": mu+c, "mu_next_measured_flux": mu+c}]
    result = failure_audit(parameters, [{**base, "status": status}], paths, residuals)[0]
    assert status.startswith(reason)
    assert result["V_next_fitted_negative_count"] == observed_bad
    assert result["V_next_measured_M3_negative_count"] == 0
    assert result["mu_next_fitted_negative_count"] == 0
    assert result["mu_last_state_outside_range"] == result["V_last_state_outside_range"] == outside
    assert result["attempted_next_V" if observed_bad else "attempted_next_mu"] < 0


def test_cli_keeps_missing_terminal_panels_and_exports_audit(tmp_path):
    rows, _ = shape_rows()
    source, output = tmp_path / "table.jsonl", tmp_path / "comparison"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, "-m", "closure_study.refine_shape", "--table", str(source), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    with (output / "metrics.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    with (output / "failure_audit.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    with (output / "trajectories.csv").open() as handle:
        paths = list(csv.DictReader(handle))
    assert all(r["mu_observed"] == r["V_observed"] == "" and r["direct_mu"] != "" for r in paths if r["step"] == "12")
    assert (output / "index.html").read_text().count("data:image/png;base64,") == 4
