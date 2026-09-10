import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from closure_study.fit_llm import fit_series
from closure_study.tests.test_fit_llm import sae_rows


def time_rows():
    rows, mu, v = [], .5, .2
    for step in range(12):
        beta, gamma = .1 + .01 * step - .001 * step**2, .2 + mu
        c, m3 = beta * v, gamma * v**1.5
        q = .8 * beta * m3
        rows.append(dict(family="llm_continuous", run_id="sae-rerun-s2", trait_id="25273", setting_id="sae",
                         inspection_status="development", step=step, step_end=step+1, interval=1,
                         mu=mu, V=v, beta=beta, C=c, M3=m3, skewness=gamma, Q=q,
                         direct_mu_old=mu, direct_mu_new=mu+c))
        mu, v = mu + c, v + q - c*c
    return rows, (mu, v)


def test_time_quadratic_recovers_selection_while_skewness_uses_generated_mean():
    rows, final = time_rows()
    parameters, metrics, paths, residuals = fit_series(rows, sae_selection=True)
    p = next(r for r in parameters if r["model"] == "flux_time_quadratic")
    np.testing.assert_allclose([p["c0"], p["c1"], p["c2"]], [.1, .01, -.001], atol=1e-12)
    for p in parameters:
        np.testing.assert_allclose([p["gamma0"], p["gamma1"], p["kappa"]], [.2, 1., .8], atol=1e-12)
        assert (p["c2"] is not None) == p["model"].endswith("quadratic")
    m = next(r for r in metrics if r["model"] == "flux_time_quadratic")
    assert m["status"] == "complete" and m["mu_rmse"] < 1e-12 and m["V_rmse"] < 1e-12
    last = next(r for r in paths if r["model"] == m["model"] and r["step"] == 12)
    np.testing.assert_allclose([last["mu_generated"], last["V_generated"]], final, atol=1e-12)
    for r in residuals:
        assert r["Q_flux_residual"] + r["Q_moment_residual"] + r["Q_selection_residual"] == pytest.approx(r["Q_total_residual"], abs=1e-14)
        if r["model"] == m["model"]:
            assert r["delta_V_residual"] == pytest.approx(0., abs=1e-14)


def test_flux_objective_weights_beta_errors_by_variance_squared():
    rows, _ = time_rows()
    rows[7]["V"] = .001
    rows[7]["beta"] = 100.
    rows[7]["C"] = .1
    _, _, _, residuals = fit_series(rows, sae_selection=True)
    for model in {r["model"] for r in residuals}:
        rs = [r for r in residuals if r["model"] == model]
        if model == "beta_ols_affine":
            assert all(r["beta_fit_weight"] == 1 for r in rs)
        else:
            assert sum(r["C_residual"]**2 for r in rs) == pytest.approx(sum(r["beta_fit_weight"] * r["beta_residual"]**2 for r in rs))
            assert all(r["beta_fit_weight"] == r["V"]**2 for r in rs)
    errors = {model: sum(r["C_residual"]**2 for r in residuals if r["model"] == model)
              for model in ("beta_ols_affine", "flux_state_affine", "flux_state_quadratic")}
    assert errors["flux_state_quadratic"] <= errors["flux_state_affine"] < errors["beta_ols_affine"]


def test_reference_unchanged_and_zero_variance_stays_in_observations():
    rows = sae_rows()
    rows[5].update(V=0., C=0., Q=0., M3=0., beta=None, skewness=None)
    before = fit_series(rows)
    parameters, metrics, paths, residuals = fit_series(rows, sae_selection=True)
    old = before[0][1]
    assert all([p[k] for k in ("gamma0", "gamma1", "kappa")] == [old[k] for k in ("gamma0", "gamma1", "kappa")] for p in parameters)
    assert all(p["fit_points"] == 11 for p in parameters)
    assert all(r["step"] != 5 for r in residuals)
    assert all(r["V_observed"] == 0 for r in paths if r["step"] == 5)
    ref = next(r for r in metrics if r["model"] == "beta_ols_affine")
    assert {k: v for k, v in ref.items() if k != "model"} == {k: v for k, v in before[1][1].items() if k != "model"}


def test_cli_embeds_comparison_and_keeps_missing_terminal_moments(tmp_path):
    rows, _ = time_rows()
    source, output = tmp_path / "table.jsonl", tmp_path / "comparison"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, "-m", "closure_study.refine_sae", "--table", str(source), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    with (output / "metrics.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 5
    with (output / "trajectories.csv").open() as handle:
        paths = list(csv.DictReader(handle))
    assert all(r["mu_observed"] == r["V_observed"] == "" and r["direct_mu"] != "" for r in paths if r["step"] == "12")
    assert (output / "index.html").read_text().count("data:image/png;base64,") == 4
