import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from closure_study.fit_controlled import continuous_residuals, fit_run, rollout


def binary_rows():
    rows, t = [], .2
    for step in range(12):
        v = t * (1 - t)
        s = .2 - .1 * t + .05 * t * t
        next_t = t + v * s
        rows.append(dict(family="tabular_binary", run_id="binary-seed-1", trait_id="trait",
                         inspection_status="development", step=step, step_end=step + 1, interval=1,
                         mu=t, T=t, V=v, S=s, C=v * s, mu_next=next_t, V_next=next_t * (1 - next_t)))
        t = next_t
    return rows


def neural_rows():
    # Independent three-point population; linear reweighting makes Q = beta*M3 exact.
    rows, z, p = [], np.array([-1., 0., 2.]), np.array([.2, .5, .3])
    for step in range(12):
        mu = p @ z
        v, m3 = p @ (z - mu) ** 2, p @ (z - mu) ** 3
        b = .01 + .02 * mu
        next_p = p * (1 + b * (z - mu))
        next_mu = next_p @ z
        rows.append(dict(family="neural_continuous", run_id="neural-seed-1", trait_id="trait",
                         inspection_status="development", step=step, step_end=step + 1, interval=1,
                         mu=mu, V=v, beta=b, C=next_mu - mu, M3=m3, skewness=m3 / v ** 1.5,
                         Q=(next_p - p) @ (z - mu) ** 2,
                         mu_next=next_mu, V_next=next_p @ (z - next_mu) ** 2))
        p = next_p
    del rows[-1]["Q"], rows[-1]["V_next"]
    return rows


def test_quadratic_state_recovered_and_time_rival_has_same_parameter_count():
    coefficients, metrics, trajectories, _ = fit_run(binary_rows())
    curves = {r["curve"]: r for r in coefficients}
    np.testing.assert_allclose([curves["state_quadratic"][f"c{i}"] for i in range(3)], [.2, -.1, .05], atol=1e-12)
    for degree in ("affine", "quadratic"):
        assert curves[f"state_{degree}"]["parameter_count"] == curves[f"time_{degree}"]["parameter_count"]
    assert len(curves) == 5  # One shared constant model, no duplicate time constant.
    best = next(r for r in metrics if r["model"] == "state_quadratic")
    assert best["status"] == "complete" and best["mu_trajectory_rmse"] < 1e-12
    expected = binary_rows()[-1]["mu_next"]
    assert next(r for r in trajectories if r["model"] == "state_quadratic" and r["step"] == 12)["mu_generated"] == pytest.approx(expected)


def test_rollouts_use_generated_states_and_discrete_variance_correction():
    binary, status = rollout([.2], [0, 1], [.1, .01, .02], time=True)
    np.testing.assert_allclose(binary[:, 0], [.2, .216, .23801472])
    assert status == "complete"
    path, status = rollout([0., 1.], [0, 1], [.1, .2], [.3, .4])
    np.testing.assert_allclose(path[:2], [[0., 1.], [.1, 1.02]])
    np.testing.assert_allclose(path[2], [.2224, 1.02 + .12 * .34 * 1.02 ** 1.5 - .1224 ** 2])
    assert status == "complete"


def test_continuous_residual_components_sum_and_terminal_variance_stays_missing():
    rows = neural_rows()
    r = continuous_residuals(rows, [.03, -.01], [.2, .1])
    np.testing.assert_allclose(r["Q_linear_residual"][:-1], 0, atol=1e-14)
    np.testing.assert_allclose(r["Q_linear_residual"] + r["Q_moment_residual"] + r["Q_selection_residual"],
                               r["Q_total_residual"], atol=1e-14)
    _, metrics, trajectories, _ = fit_run(rows)
    assert len(metrics) == 4 and all(r["status"] == "complete" for r in metrics)
    assert all(np.isnan(r["V_observed"]) for r in trajectories if r["step"] == 12)
    assert all(np.isfinite(r["mu_observed"]) for r in trajectories if r["step"] == 12)


@pytest.mark.parametrize("initial,selection,skewness,reason", [
    ([.9], [2.], None, "invalid_probability"), ([0., 1.], [2.], [0.], "negative_variance")])
def test_invalid_rollouts_stop_without_clipping(initial, selection, skewness, reason):
    path, status = rollout(initial, [0, 1], selection, skewness)
    assert status.startswith(reason) and np.isnan(path[1:]).all()


def test_nonunit_intervals_and_holdout_runs_rejected():
    rows = binary_rows()
    rows[0]["interval"] = 5
    with pytest.raises(ValueError, match="one-update"):
        fit_run(rows)
    rows = binary_rows()
    rows[0]["inspection_status"] = "holdout"
    with pytest.raises(ValueError, match="development"):
        fit_run(rows)


def test_cli_exports_both_systems_with_embedded_report(tmp_path):
    source = tmp_path / "transitions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in binary_rows() + neural_rows()))
    before = source.read_bytes()
    output = tmp_path / "fits"
    command = [sys.executable, "-m", "closure_study.fit_controlled", "--table", str(source), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert source.read_bytes() == before
    with (output / "metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 9 and all(r["status"] == "complete" for r in metrics)
    assert (output / "index.html").read_text().count("data:image/png;base64,") == 4
    with (output / "trajectories.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(r["V_observed"] == "" for r in rows if r["family"] == "neural_continuous" and r["step"] == "12")
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "new or empty" in again.stderr
