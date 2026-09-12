"""Small per-run closure comparison on the two controlled development systems."""

import argparse
import base64
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

from .io import read_jsonl, write_csv


BINARY = [("constant", "T", 0), ("state_affine", "T", 1), ("state_quadratic", "T", 2),
          ("time_affine", "step", 1), ("time_quadratic", "step", 2)]
CONTINUOUS = [(f"beta{b}_gamma{g}", b, g) for b in (0, 1) for g in (0, 1)]


def values(rows, key):
    return np.asarray([r.get(key, np.nan) for r in rows], dtype=float)


def rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else None


def fit_curve(x, y, degree):
    valid = np.isfinite(x) & np.isfinite(y)
    if len(np.unique(x[valid])) <= degree:
        raise ValueError("not enough distinct measured states/times for requested degree")
    return polyfit(x[valid], y[valid], degree)


def selection_value(selection, mu, v, step, predictor="mu"):
    if predictor == "mu_V":
        return selection[0] + selection[1] * mu + selection[2] * v
    return polyval({"mu": mu, "T": mu, "step": step}[predictor], selection)


def rollout(initial, steps, selection, skewness=None, time=False, *, kappa=1.0, mean_bounds=None, skewness_time=False,
            selection_variance=False):
    """Only the initial state and fitted laws enter the discrete recurrence."""
    path = np.full((len(steps) + 1, len(initial)), np.nan)
    path[0] = initial
    for i, step in enumerate(steps):
        mu = path[i, 0]
        v = path[i, 1] if skewness is not None else mu * (1 - mu)
        with np.errstate(over="ignore", invalid="ignore"):
            b = selection_value(selection, mu, v, step, "mu_V" if selection_variance else "step" if time else "mu")
            c = b * v
            next_state = [mu + c]
            if skewness is not None:
                q = kappa * b * polyval(step if skewness_time else mu, skewness) * v ** 1.5
                next_state.append(v + q - c * c)
        if not np.isfinite(next_state).all():
            return path, f"nonfinite_state_at_{step + 1}"
        if skewness is None and not 0 <= next_state[0] <= 1:
            return path, f"invalid_probability_at_{step + 1}"
        if skewness is not None and next_state[1] < 0:
            return path, f"negative_variance_at_{step + 1}"
        if mean_bounds is not None and not mean_bounds[0] <= next_state[0] <= mean_bounds[1]:
            return path, f"mean_outside_support_at_{step + 1}"
        path[i + 1] = next_state
    return path, "complete"


def continuous_residuals(rows, selection, skewness):
    """Separate linear reweighting, moment closure, and selection closure errors."""
    mu, v, b, m3, q = (values(rows, k) for k in ("mu", "V", "beta", "M3", "Q"))
    bhat, mhat = polyval(mu, selection), polyval(mu, skewness) * v ** 1.5
    return {"C_residual": values(rows, "C") - bhat * v,
            "Q_linear_residual": q - b * m3,
            "Q_moment_residual": b * (m3 - mhat),
            "Q_selection_residual": (b - bhat) * mhat,
            "Q_total_residual": q - bhat * mhat,
            "delta_V_residual": q - values(rows, "C") ** 2 - (bhat * mhat - (bhat * v) ** 2)}


def validate_run(rows):
    steps = values(rows, "step")
    if (not np.isfinite(steps).all() or (steps != np.floor(steps)).any()
            or (np.diff(steps) != 1).any() or (values(rows, "step_end") != steps + 1).any()
            or (values(rows, "interval") != 1).any()):
        raise ValueError("controlled fits require contiguous, one-update transitions")
    if any(r.get("inspection_status") != "development" for r in rows):
        raise ValueError("controlled fitting is restricted to development runs")
    mu, v = values(rows, "mu"), values(rows, "V")
    if not np.isfinite(mu).all() or not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("this comparison requires finite means and positive measured variances")
    if any(r.get("support_failure") for r in rows):
        raise ValueError("stored support failures must be resolved before fitting")
    return steps, mu, v


def fit_run(rows):
    steps, mu, v = validate_run(rows)
    binary = rows[0]["family"] == "tabular_binary"
    observed = np.column_stack([np.r_[mu, rows[-1]["mu_next"]],
                                np.r_[v, rows[-1].get("V_next", np.nan)]])
    base = {k: rows[0][k] for k in ("family", "run_id", "trait_id")}
    curves, coefficients, metrics, trajectories, residuals = {}, [], [], [], []
    specs = [(name, x, "S", d) for name, x, d in BINARY] if binary else [
        (f"{target}{d}", "mu", "skewness" if target == "gamma" else "beta", d)
        for target in ("beta", "gamma") for d in (0, 1)]
    for name, predictor, target, degree in specs:
        x, y = values(rows, predictor), values(rows, target)
        coef = fit_curve(x, y, degree)
        curves[name] = coef
        coefficients.append({**base, "curve": name, "target": target, "predictor": predictor,
                             "degree": degree, "parameter_count": degree + 1,
                             "c0": coef[0], "c1": coef[1] if degree >= 1 else None,
                             "c2": coef[2] if degree == 2 else None,
                             "n_fit": int((np.isfinite(x) & np.isfinite(y)).sum()),
                             "coefficient_rmse": rms(y - polyval(x, coef))})
    for name, x, degree in BINARY if binary else CONTINUOUS:
        selection = curves[name] if binary else curves[f"beta{x}"]
        skewness = None if binary else curves[f"gamma{degree}"]
        path, status = rollout(observed[0, :1 if binary else 2], steps.astype(int), selection,
                               skewness, time=binary and x == "step")
        if binary:
            r = {"C_residual": values(rows, "C") - polyval(values(rows, x), selection) * v}
        else:
            r = continuous_residuals(rows, selection, skewness)
        metric = {**base, "model": name, "status": status, "n_updates": len(rows),
                  "completed_updates": int(np.isfinite(path[1:, 0]).sum()),
                  "mu_trajectory_rmse": rms(path[1:, 0] - observed[1:, 0]) if status == "complete" else None,
                  "V_trajectory_rmse": rms(path[1:, 1] - observed[1:, 1]) if not binary and status == "complete" else None,
                  "C_rmse": rms(r["C_residual"])}
        if not binary:
            measured_q = np.isfinite(values(rows, "Q"))
            q_scale = rms(values(rows, "Q")[measured_q])
            for key in r:
                if key.startswith("Q_"):
                    metric[key + "_relative_l2"] = rms(r[key][measured_q]) / q_scale if q_scale else None
            metric["delta_V_rmse"] = rms(r["delta_V_residual"])
        metrics.append(metric)
        for i, step in enumerate(np.r_[steps, steps[-1] + 1]):
            trajectories.append({**base, "model": name, "step": int(step), "mu_observed": observed[i, 0],
                                 "mu_generated": path[i, 0], "V_observed": observed[i, 1] if not binary else None,
                                 "V_generated": path[i, 1] if not binary else None})
        for i, step in enumerate(steps):
            residuals.append({**base, "model": name, "step": int(step), **{k: a[i] for k, a in r.items()}})
    return coefficients, metrics, trajectories, residuals


def table(records, fields):
    def cell(value):
        return escape(f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value if value is not None else "—"))
    return "<table><tr>" + "".join(f"<th>{escape(k)}</th>" for k in fields) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{cell(r.get(k))}</td>" for k in fields) + "</tr>" for r in records) + "</table>"


def report(output, groups, coefficients, metrics, trajectories):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=125, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        figures.append(f"<figure><img src='data:image/png;base64,{encoded}' alt='{name}'><figcaption>{caption}</figcaption></figure>")

    def coefficients_for(rid, name):
        r = next(c for c in coefficients if c["run_id"] == rid and c["curve"] == name)
        return [r[f"c{i}"] for i in range(r["degree"] + 1)]

    def errors(ax, family, field, models):
        runs = sorted({r["run_id"] for r in metrics if r["family"] == family})
        for name in models:
            sub = [next(r for r in metrics if r["run_id"] == rid and r["model"] == name) for rid in runs]
            ax.plot(range(len(runs)), [r.get(field) if r.get(field) is not None else np.nan for r in sub], "o-", label=name, ms=4)
        ax.set(xticks=range(len(runs)), xticklabels=[rid.split("-")[-1] for rid in runs], xlabel="Recorded seed", ylabel=field)
        ax.grid(alpha=.2)

    def generated(ax, rid, state, models):
        for name in models:
            sub = [r for r in trajectories if r["run_id"] == rid and r["model"] == name]
            ax.plot([r["step"] for r in sub], [r[f"{state}_generated"] for r in sub], label=name, lw=1.2)
        ax.plot([r["step"] for r in sub], [r[f"{state}_observed"] for r in sub], "k--", label="observed", lw=1.4)
        ax.set(xlabel="Optimizer step", ylabel="T" if state == "mu" and "tabular" in rid else state)
        ax.grid(alpha=.2)

    for family in ("tabular_binary", "neural_continuous"):
        key = next(k for k in sorted(groups) if k[0] == family)
        series, rid = groups[key], key[1]
        binary = family == "tabular_binary"
        panels = [("T", "S"), ("step", "S")] if binary else [("mu", "beta"), ("mu", "skewness")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, (x, y) in zip(axes, panels):
            ax.scatter(values(series, x), values(series, y), s=5, alpha=.3, color="grey", label="measured")
            grid = np.linspace(values(series, x).min(), values(series, x).max(), 200)
            names = (["constant"] + [name for name, px, d in BINARY if d and px == x]) if binary else [f"{'gamma' if y == 'skewness' else 'beta'}{d}" for d in (0, 1)]
            for name in names:
                ax.plot(grid, polyval(grid, coefficients_for(rid, name)), label=name)
            ax.set(xlabel=x, ylabel=y, title=rid)
            ax.legend(fontsize=8)
        save(fig, f"{family}_fits", "First recorded run, chosen by run ID. All runs' coefficients are tabulated below. Time uses raw optimizer steps; 0/1 in continuous names denotes polynomial degree.")
        models = [m[0] for m in (BINARY if binary else CONTINUOUS)]
        fig, axes = plt.subplots(1 if binary else 2, 2, figsize=(11, 4 if binary else 7), squeeze=False)
        for axs, state in zip(axes, ["mu"] if binary else ["mu", "V"]):
            generated(axs[0], rid, state, models)
            errors(axs[1], family, f"{state}_trajectory_rmse", models)
            axs[0].legend(fontsize=7)
        save(fig, f"{family}_trajectories", "Left: initial-state-only generation for the first recorded run. Right: separate full-horizon reconstruction errors for every run. Missing errors indicate failed trajectories, never successful short prefixes.")

    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Controlled closure fits</title>",
            "<style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui;color:#17212b}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{border-collapse:collapse;font-size:.8rem;display:block;overflow:auto}th,td{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>Controlled closure fits: development reconstruction</h1>",
            "<p>Every coefficient is fitted within one run by unweighted least squares on measured S, β, or standardized skewness γ. These are fits to coefficients, not optimizations of trajectory error. All saved finite coefficient observations are used. No coefficients are pooled and no held-out performance is claimed.</p>",
            "<p>Binary candidates: a common constant S, affine and quadratic S(T), and matched affine and quadratic S(t). The constant is identical for state and time. Continuous candidates combine constant/affine β(μ) with constant/affine γ(μ), where M₃ = γV<sup>3/2</sup>. Model beta1_gamma0, for example, uses affine β and constant γ.</p>",
            "<p>Generation uses T′ = T + T(1−T)S, or μ′ = μ + βV and V′ = V + βγV<sup>3/2</sup> − (βV)². Each law is evaluated on the generated state, or the clock for a time rival. Only the initial state is supplied. Invalid probabilities, negative variance, or nonfinite states stop a trajectory; values are never clipped. The same-run fit makes these development reconstructions, not prospective predictions.</p>",
            "<p>Trajectory RMSE excludes the supplied initial state. The missing final neural variance is omitted from its score; the final mean is retained. Coefficients use raw T, μ, or optimizer step: c0 + c1 x + c2 x². Display rounding does not affect calculation; CSVs retain full precision.</p>",
            "<p>Continuous Q residuals are separated as: Q−βM₃ (linearity), β(M₃−M̂₃) (moment closure), and (β−β̂)M̂₃ (selection closure). Their signed sum equals Q−β̂M̂₃. The relative L2 columns divide each component's norm by the measured Q norm on the same available transitions. Component norms do not add; cancellation is possible. These are descriptive residuals, not uncertainty estimates or significance tests.</p>"]
    html.extend(figures)
    for (family, rid, trait), series in sorted(groups.items()):
        cs, ms = ([r for r in records if r["run_id"] == rid and r["trait_id"] == trait] for records in (coefficients, metrics))
        html.append(f"<details><summary>{escape(rid)} · trait {escape(trait)}</summary>")
        html.append(table(cs, ["curve", "target", "predictor", "parameter_count", "c0", "c1", "c2", "coefficient_rmse", "n_fit"]))
        html.append(table(ms, ["model", "status", "mu_trajectory_rmse", "V_trajectory_rmse", "C_rmse"] +
                          ([f"Q_{part}_residual_relative_l2" for part in ("linear", "moment", "selection", "total")] if family == "neural_continuous" else [])))
        html.append("</details>")
    html.append("<p>Companion CSVs: coefficients, metrics, trajectories, and signed per-transition residuals. The four figures are also saved as PNG/PDF. This HTML embeds all figures. Compare interpretation, residuals, and trajectories together; no model is automatically selected. Stop for discussion before transfer tests, new models, or new training.</p></html>")
    (output / "index.html").write_text("\n".join(html))


def run(table_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    groups = defaultdict(list)
    for row in read_jsonl(table_path):
        if row["family"] in ("tabular_binary", "neural_continuous"):
            groups[(row["family"], row["run_id"], row["trait_id"])].append(row)
    if {k[0] for k in groups} != {"tabular_binary", "neural_continuous"}:
        raise ValueError("supply the diagnostic table containing both controlled systems")
    combined = [[], [], [], []]
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["step"])
        for target, records in zip(combined, fit_run(rows)):
            target.extend(records)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in zip(("coefficients", "metrics", "trajectories", "residuals"), combined):
        clean = [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                  for k, v in r.items()} for r in records]
        write_csv(output / f"{name}.csv", clean)
    report(output, groups, *combined[:3])
    print(f"Fitted {len(groups)} individual runs; generated {len(combined[1])} trajectories. Report: {output / 'index.html'}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True, help="existing diagnostic transitions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.table, args.output)
