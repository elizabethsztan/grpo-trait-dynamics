import copy
import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from closure_study.fit_controlled import rollout
from closure_study.statistics import moments
from closure_study.transfer_controlled import (binary_transfer, continuous_transfer, early_diagnostics,
                                               fit_kappa, kappa_stability, trait_bins, upper_root)
from closure_study.tests.test_fit_controlled import binary_rows, neural_rows


def runs(factory):
    result = {}
    for seed in (1, 2):
        rows = factory()
        rid = rows[0]["run_id"].rsplit("-", 1)[0] + f"-{seed}"
        result[rid] = [{**r, "run_id": rid} for r in rows]
    return result


def test_kappa_is_through_origin_and_scales_only_Q():
    rows = [{"beta": x, "M3": 1., "Q": .8 * x} for x in (-2., 1., 3.)]
    assert fit_kappa(rows) == pytest.approx(.8)
    path, status = rollout([0., 1.], [0], [.1], [.3], kappa=.8)
    np.testing.assert_allclose(path[1], [.1, 1.014])
    assert status == "complete"
    with pytest.raises(ValueError, match="unidentified"):
        fit_kappa([{"beta": 0., "M3": 1., "Q": 0.}])


def test_binary_omitted_future_cannot_change_loo_coefficients_or_generation():
    data = runs(binary_rows)
    rid = sorted(data)[0]
    before = binary_transfer(data)
    changed = copy.deepcopy(data)
    for r in changed[rid][1:]:
        r["S"] *= 2
        r["mu"] += .01
        r["T"] += .01
    after = binary_transfer(changed)
    for model in ("state_loo", "time_loo"):
        params = [next(r for r in result[0] if r["run_id"] == rid and r["model"] == model) for result in (before, after)]
        assert params[0] == params[1] and rid not in params[0]["training_runs"]
        paths = [[r["mu_generated"] for r in result[2] if r["run_id"] == rid and r["model"] == model] for result in (before, after)]
        np.testing.assert_array_equal(*paths)
    assert upper_root([-.12, .8, -1.]) == pytest.approx(.6)


def test_continuous_full_transfer_excludes_target_and_components_sum():
    data = runs(neural_rows)
    rid = sorted(data)[0]
    before = continuous_transfer(data)
    changed = copy.deepcopy(data)
    for r in changed[rid][1:]:
        r["beta"] *= 1.2
        r["skewness"] *= .9
        if "Q" in r:
            r["Q"] *= .8
    after = continuous_transfer(changed)
    params = [next(r for r in result[0] if r["run_id"] == rid and r["model"] == "loo_all") for result in (before, after)]
    assert params[0] == params[1]
    assert rid not in params[0]["beta_gamma_training_runs"] and rid not in params[0]["kappa_training_runs"]
    paths = [[(r["mu_generated"], r["V_generated"]) for r in result[2] if r["run_id"] == rid and r["model"] == "loo_all"] for result in (before, after)]
    np.testing.assert_array_equal(*paths)
    partial = next(r for r in before[0] if r["run_id"] == rid and r["model"] == "loo_kappa")
    assert partial["beta_gamma_training_runs"] == [rid] and rid not in partial["kappa_training_runs"]
    for r in after[3]:
        assert r["Q_flux_residual"] + r["Q_moment_residual"] + r["Q_selection_residual"] == pytest.approx(r["Q_total_residual"], abs=1e-14)


def test_trait_bins_sum_exact_flux_with_variation_inside_bins():
    z, p, q = np.array([-2., -1., -.9, 1., 3.]), np.array([.1, .2, .3, .25, .15]), np.array([.08, .18, .26, .30, .18])
    state = moments(z, p)
    x = z - state["mu"]
    beta, Q = ((q - p) @ x) / state["V"], (q - p) @ (x * x)
    bins = trait_bins(z, p, q, beta, .8)
    assert len(bins) < len(z)  # At least two distinct trait values share a bin.
    assert sum(r["Q_linear_residual"] for r in bins) == pytest.approx(Q - beta * state["M3"], abs=1e-14)
    assert sum(r["Q_flux_residual"] for r in bins) == pytest.approx(Q - .8 * beta * state["M3"], abs=1e-14)
    assert sum(r["mass"] for r in bins) == pytest.approx(1.)


def archive_fixture(path):
    z, p, weights = np.array([-1., 0., 2.]), np.array([.2, .5, .3]), []
    for _ in range(12):
        weights.append(p)
        mu = p @ z
        p = p * (1 + (.01 + .02 * mu) * (z - mu))
    np.savez(path, seeds=[1, 2], particle_scores=[z, z], particle_weights=[weights, weights])


def test_early_archive_alignment_and_window_accounting(tmp_path):
    data = runs(neural_rows)
    archive = tmp_path / "particles.npz"
    archive_fixture(archive)
    bins = early_diagnostics(archive, data)
    assert {r["step"] for r in bins} == {0, 10}
    windows = kappa_stability(data)
    assert all(r["fit_weight_share"] == pytest.approx(1.) for r in windows)
    data[sorted(data)[0]][0]["Q"] += .1
    with pytest.raises(ValueError, match="do not match table"):
        early_diagnostics(archive, data)


def test_cli_transfer_report_and_missing_final_variance(tmp_path):
    data = {**runs(binary_rows), **runs(neural_rows)}
    source, archive, output = tmp_path / "table.jsonl", tmp_path / "particles.npz", tmp_path / "out"
    source.write_text("".join(json.dumps(r) + "\n" for rows in data.values() for r in rows))
    archive_fixture(archive)
    subprocess.run([sys.executable, "-m", "closure_study.transfer_controlled", "--table", str(source),
                    "--neural-archive", str(archive), "--output", str(output)], check=True, capture_output=True, text=True)
    with (output / "metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 18 and all(r["status"] == "complete" for r in metrics)
    assert (output / "index.html").read_text().count("data:image/png;base64,") == 4
    with (output / "trajectories.csv").open() as handle:
        trajectories = list(csv.DictReader(handle))
    assert all(r["V_observed"] == "" for r in trajectories if r["family"] == "neural_continuous" and r["step"] == "12")
