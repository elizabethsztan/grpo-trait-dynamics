import copy
import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from closure_study.fit_controlled import rollout
from closure_study.fit_llm import fit_binary_flux, fit_series, observations
from closure_study.tests.test_fit_controlled import binary_rows, neural_rows


def output_rows():
    rows = binary_rows()
    for r in rows:
        r.update(family="llm_binary", setting_id="output-above_hook", run_id="output-1")
        if r["step"] % 3:
            r["T"] = None
        if r["step_end"] % 3:
            r["mu_next"] = None
    return rows


def sae_rows():
    rows = neural_rows()
    for r in rows:
        r.update(family="llm_continuous", setting_id="sae", run_id="sae-rerun-s2", trait_id="25273")
        # Translate the finite population to nonnegative support; central moments do not change.
        r["mu"] += 1
        r["direct_mu_old"] = r["mu"] + .01
        r["direct_mu_new"] = r.pop("mu_next") + 1.01
        r.pop("V_next", None)
        r["Q"] = .8 * r["beta"] * r["M3"]
    return rows


def test_flux_fit_handles_boundaries_without_dividing_by_variance():
    rows = [{"T": t, "C": t * (1 - t) * (.2 - .1 * t + .05 * t*t)}
            for t in [0., .1, .3, .6, .9, 1.]]
    rows[0]["C"], rows[-1]["C"] = 100., -100.  # Zero information about coefficients at boundaries.
    coef, valid, count = fit_binary_flux(rows, "T", 2)
    np.testing.assert_allclose(coef, [.2, -.1, .05], atol=1e-12)
    assert valid.all() and count == 4
    with pytest.raises(ValueError, match="unidentified"):
        fit_binary_flux([rows[0], rows[-1]], "T", 0)


def test_sparse_binary_fits_and_scores_only_measured_checkpoints():
    rows = output_rows()
    before = fit_series(rows)
    curves = {r["model"]: r for r in before[0]}
    np.testing.assert_allclose([curves["state_quadratic"][f"c{i}"] for i in range(3)], [.2, -.1, .05], atol=1e-10)
    for name in ["state_quadratic", "time_quadratic"]:
        assert all(curves[name][f"c{i}"] is not None for i in range(3))
    assert all(r["fit_points"] == 4 for r in before[0])
    assert all(r["mu_score_points"] == 4 for r in before[1])
    assert next(r for r in before[1] if r["model"] == "state_quadratic")["mu_rmse"] < 1e-12
    for r in rows:
        if r["T"] is None:
            r["C"] = 1000.
    assert fit_series(rows)[0] == before[0]
    observed = observations(rows, True)[:, 0]
    assert np.isfinite(observed).sum() == 5 and np.isnan(observed[1:3]).all()


def test_sae_separate_observations_and_flux_residual_decomposition():
    rows = sae_rows()
    parameters, metrics, paths, residuals = fit_series(rows)
    assert parameters[1]["kappa"] == pytest.approx(.8)
    assert all(r["mu_score_points"] == 11 and r["direct_score_points"] == 12 for r in metrics)
    assert all(np.isnan(r["mu_observed"]) and np.isnan(r["V_observed"]) and np.isfinite(r["direct_mu"])
               for r in paths if r["step"] == 12)
    for r in residuals:
        assert r["Q_flux_residual"] + r["Q_moment_residual"] + r["Q_selection_residual"] == pytest.approx(r["Q_total_residual"], abs=1e-14)
    changed = copy.deepcopy(rows)
    for r in changed:
        del r["direct_mu_old"], r["direct_mu_new"]
        r["direct_delta"] = 999.
    assert np.isnan(observations(changed, False)[:, 2]).all()
    np.testing.assert_array_equal([r["mu_generated"] for r in paths], [r["mu_generated"] for r in fit_series(changed)[2]])


def test_zero_variance_panel_is_excluded_from_fits_but_retained_as_observation():
    rows = sae_rows()
    rows[5].update(V=0., C=0., Q=0., M3=0., beta=None, skewness=None)
    parameters, metrics, paths, residuals = fit_series(rows)
    assert all(r["fit_points"] == r["informative_points"] == 11 for r in parameters)
    assert parameters[1]["kappa"] == pytest.approx(.8)
    assert len(residuals) == 22 and all(r["step"] != 5 for r in residuals)
    for m in metrics:
        assert m["Q_linear_residual_relative_l2"] == pytest.approx(.25)
        assert m["mu_score_points"] == 11 and m["direct_score_points"] == 12
        scored = [r for r in paths if r["model"] == m["model"] and 0 < r["step"] < 12]
        assert scored[4]["V_observed"] == 0.
        expected = np.sqrt(np.mean([(r["V_generated"] - r["V_observed"]) ** 2 for r in scored]))
        assert m["V_rmse"] == pytest.approx(expected)


def test_support_stops_sae_mean_but_allows_signed_controlled_traits():
    path, status = rollout([.1, 1.], [0, 1], [-.2], [0.], mean_bounds=(0., np.inf))
    assert status == "mean_outside_support_at_1" and np.isnan(path[1:]).all()
    path, status = rollout([.1, 1.], [0, 1], [-.2], [0.])
    assert status == "complete" and path[1, 0] < 0


def test_cli_embeds_report_and_leaves_missing_observations_empty(tmp_path):
    rows = output_rows() + [{**r, "run_id": "output-2", "setting_id": "output-all-layers"} for r in output_rows()] + sae_rows()
    source, output = tmp_path / "table.jsonl", tmp_path / "fits"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, "-m", "closure_study.fit_llm", "--table", str(source), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    with (output / "metrics.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 12
    with (output / "trajectories.csv").open() as handle:
        paths = list(csv.DictReader(handle))
    assert all(r["mu_observed"] == r["V_observed"] == "" for r in paths if r["family"] == "llm_continuous" and r["step"] == "12")
    assert (output / "index.html").read_text().count("data:image/png;base64,") == 4
