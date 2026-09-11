import numpy as np
import pytest

from closure_study.diagnose_consistency import diagnose, event_summary, moment_margin, read_export, run
from closure_study.fit_llm import fit_series
from closure_study.io import read_csv, write_csv
from closure_study.tests.test_refine_shape import shape_rows


def test_nonnegative_moment_condition_and_trait_rescaling():
    # A two-point nonnegative distribution saturates the bound.
    z, p = np.array([0., 3.]), np.array([.8, .2])
    mu = p @ z
    v, m3 = p @ (z - mu)**2, p @ (z - mu)**3
    assert moment_margin(mu, v, m3)[1] == pytest.approx(0., abs=1e-15)
    assert moment_margin(1., 1., 2.)[0] == 2.  # Exponential moments.
    assert moment_margin(1., 1., -1.)[1] < 0
    assert moment_margin(0., 0., 0.) == (0., 0.)
    assert np.isnan(moment_margin(np.nan, np.nan, np.nan)[1])
    for scale in (1e-6, 100.):
        raw, relative = moment_margin(scale, scale**2, -scale**3)
        assert raw == pytest.approx(-scale**4, abs=0., rel=1e-12)
        assert relative == pytest.approx(moment_margin(1., 1., -1.)[1])


@pytest.mark.parametrize("crash", [False, True])
def test_event_order_preserves_source_time_missingness_and_completion(crash):
    rows = [dict(step=t, measured_relative_margin=0., fitted_relative_margin=0.,
                 generated_relative_margin=0. if t == 0 else -.1,
                 V_relative_error=[0., .15, .3, np.nan][t], next_mu=1., next_V=1.) for t in range(4)]
    if crash:
        rows[1]["next_mu"] = -.1  # Same source as the first moment violation.
        rows[2].update(generated_relative_margin=np.nan, V_relative_error=np.nan, next_mu=np.nan, next_V=np.nan)
    rows[3].update(next_mu=np.nan, next_V=np.nan)  # Never inspect an update after the requested horizon.
    if crash:
        rows[3]["generated_relative_margin"] = np.nan
    event = event_summary({}, "mean_outside_support_at_2" if crash else "complete", [-.1, 1., 0.], rows)
    assert event["generated_first_violation_step"] == 1
    assert event["negative_mu_from_step"] == (1 if crash else None)
    assert event["negative_mu_to_step"] == (2 if crash else None)
    assert event["V_departure_10_step"] == 1
    assert event["V_departure_25_step"] == (None if crash else 2)
    assert event["V_departure_50_step"] is None
    assert event["generated_checked_points"] == (2 if crash else 4)
    assert event["measured_violation_count"] == 0
    assert event["selection_roots"] == pytest.approx([.1])


def exports(tmp_path):
    rows, _ = shape_rows()
    rows[5].update(beta=None, skewness=None)  # Keep observed moments; remove this common fitting row.
    records = fit_series(rows, sae_shape=True)
    source = tmp_path / "fits"
    source.mkdir()
    names = ("parameters", "metrics", "trajectories", "residuals")
    for name, rs in zip(names, records):
        write_csv(source / f"{name}.csv", rs)
    return source, [read_export(source / f"{name}.csv") for name in names]


def test_saved_clocks_masks_terminal_states_and_window_denominators(tmp_path):
    _, records = exports(tmp_path)
    points, events, windows = diagnose(*records)
    model = "moment_time_quadratic"
    ps = [r for r in points if r["model"] == model]
    p = next(r for r in records[0] if r["model"] == model)
    for r in ps:
        gamma = .2 + .02 * r["step"] - .001 * r["step"]**2
        assert r["M3_generated"] == pytest.approx(gamma * r["V_generated"]**1.5)
        if r["step"] in (5, 12):
            assert np.isnan(r["measured_margin"]) and np.isnan(r["fitted_margin"])
        else:
            assert np.isfinite(r["measured_margin"])
    assert np.isnan(ps[-1]["next_mu"]) and np.isnan(ps[-1]["V_observed"])
    assert np.isfinite(ps[-1]["direct_mu"]) and np.isfinite(ps[-1]["generated_margin"])
    assert np.isfinite(ps[5]["V_observed"]) and np.isfinite(ps[5]["V_relative_error"])
    late = next(r for r in windows if r["model"] == model and r["window"] == "late")
    assert late["points"] == 6 and late["step_min"] == 6  # Split at clock 5; do not move it to replace the missing row.
    rs = [r for r in records[3] if r["model"] == model and r["step"] >= 5]
    q = np.array([r["Q_flux_residual"] + p["kappa"] * r["beta_measured"] * r["M3"] for r in rs])
    assert late["Q_total_relative_l2"] == pytest.approx(np.linalg.norm([r["Q_total_residual"] for r in rs]) / np.linalg.norm(q))
    assert sum(r["gamma_weight_share"] for r in windows if r["model"] == model) == pytest.approx(1.)


def test_command_reads_exports_without_refitting_or_modifying_them(tmp_path, monkeypatch):
    source, _ = exports(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    def no_refit(*args, **kwargs):
        raise AssertionError("diagnostics must not fit coefficients")
    monkeypatch.setattr("closure_study.fit_llm.fit_series", no_refit)
    output = tmp_path / "diagnostics"
    run(source, output)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert len(read_csv(output / "events.csv")) == 4
    assert len(read_csv(output / "windows.csv")) == 8
    assert (output / "index.html").read_text().count("data:image/png;base64,") == 3
    with pytest.raises(ValueError, match="new or empty"):
        run(source, output)
