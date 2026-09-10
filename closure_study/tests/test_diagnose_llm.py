import copy
import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from closure_study.diagnose_llm import binary_transfer, correlation, sae_diagnostics
from closure_study.tests.test_fit_llm import output_rows, sae_rows


def binary_runs():
    return {f"output-{seed}": [{**r, "run_id": f"output-{seed}"} for r in output_rows()] for seed in (1, 2, 3)}


def test_omitted_future_cannot_change_transfer_coefficients_or_generated_states():
    runs = binary_runs()
    before = binary_transfer(runs)
    changed = copy.deepcopy(runs)
    for r in changed["output-1"]:
        r["C"] *= 2
        if r["step"] > 0 and r["T"] is not None:
            r["T"] += .02
        if r["mu_next"] is not None:
            r["mu_next"] += .02
    after = binary_transfer(changed)
    def omitted(records):
        return [r for r in records if r["run_id"] == "output-1" and r["scope"] == "loo"]
    assert omitted(before[0]) == omitted(after[0])
    assert all(r["training_runs"] == ["output-2", "output-3"] and r["fit_points"] == 8 for r in omitted(after[0]))
    np.testing.assert_array_equal([r["mu_generated"] for r in omitted(before[2])],
                                  [r["mu_generated"] for r in omitted(after[2])])
    assert all(r["mu_score_points"] == r["C_score_points"] == 4 for r in omitted(after[1]))
    assert omitted(before[1]) != omitted(after[1])


@pytest.mark.parametrize("field,value,reason", [("setting_id", "different", "one setting"),
                                                ("interval", 2, "one-update"),
                                                ("inspection_status", "holdout", "development")])
def test_transfer_rejects_mixed_settings_nonunit_intervals_and_uninspected_runs(field, value, reason):
    runs = binary_runs()
    runs["output-1"][0][field] = value
    with pytest.raises(ValueError, match=reason):
        binary_transfer(runs)


def test_sae_diagnostics_keep_raw_points_and_mask_undefined_residuals():
    rows = sae_rows()
    rows[5].update(V=0., C=0., Q=0., M3=0., beta=None, skewness=None)
    points, summaries = sae_diagnostics({("sae-rerun-s2", "25273"): rows})
    assert len(points) == 12 and summaries[0]["fit_points"] == 11
    assert points[5]["V"] == 0 and "beta_residual" not in points[5]
    assert summaries[0]["Q_linear_residual_relative_l2"] == pytest.approx(.25)
    for point in points:
        if "beta_residual" in point:
            assert point["C_residual"] == pytest.approx(point["beta_residual"] * point["V"], abs=1e-14)
    assert correlation([1., 2., None], [3., 6., 999.]) == pytest.approx(1.)
    assert correlation([1., 1., 1.], [2., 3., 4.]) is None


def test_cli_exports_three_run_transfer_and_explicit_time_panel(tmp_path):
    rows = [r for run in binary_runs().values() for r in run] + sae_rows()
    source, output = tmp_path / "table.jsonl", tmp_path / "diagnostics"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    command = [sys.executable, "-m", "closure_study.diagnose_llm", "--table", str(source), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with (output / "binary_metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 18 and all(r["status"] == "complete" for r in metrics)
    with (output / "binary_trajectories.csv").open() as handle:
        paths = list(csv.DictReader(handle))
    assert all(r["mu_observed"] == "" for r in paths if int(r["step"]) % 3)
    html = (output / "index.html").read_text()
    assert html.count("data:image/png;base64,") == 4 and "beta-versus-time" in html
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "new or empty" in again.stderr
