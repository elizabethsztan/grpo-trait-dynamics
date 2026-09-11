"""Moment consistency and event ordering from saved SAE shape-study exports."""

import argparse
import base64
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyval, polyroots

from .fit_controlled import rms, table, values
from .io import read_csv, write_csv
from .refine_shape import LABELS, SAE_SHAPE
import matplotlib.pyplot as plt


PRIMARY = "moment_state_quadratic"
MOMENT_RTOL = 1e-12  # Roundoff tolerance, not a sampling-error threshold.
TEXT_FIELDS = {"run_id", "family", "setting_id", "trait_id", "model", "status", "is_control",
               "predictor", "gamma_predictor", "selection_objective", "shape_objective"}


def read_export(path):
    return [{k: v if k in TEXT_FIELDS else float(v) if v else np.nan for k, v in r.items()}
            for r in read_csv(path)]


def key(row):
    return tuple(row[k] for k in ("run_id", "trait_id", "model"))


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def moment_margin(mu, v, m3):
    """Necessary for z >= 0: E[z] E[z³] >= E[z²]². Passing is not sufficient."""
    term, spread = np.asarray(mu) * m3, np.asarray(mu) ** 2 * v
    margin = term + spread - np.asarray(v) ** 2
    scale = np.abs(term) + spread + np.asarray(v) ** 2
    relative = margin / np.where(scale == 0, 1., scale)
    return margin, relative


def first(rows, predicate):
    return next((r["step"] for r in rows if predicate(r)), None)


def event_summary(base, status, selection, rows):
    event = {**base, "status": status,
             "selection_roots": [float(z.real) for z in polyroots(selection) if abs(z.imag) < 1e-10 and z.real >= 0]}
    for name in ("measured", "fitted", "generated"):
        field = name + "_relative_margin"
        event[name + "_checked_points"] = sum(np.isfinite(r[field]) for r in rows)
        event[name + "_violation_count"] = sum(r[field] < -MOMENT_RTOL for r in rows)
        event[name + "_first_violation_step"] = first(rows, lambda r: r[field] < -MOMENT_RTOL)
    for percent in (10, 25, 50):
        event[f"V_departure_{percent}_step"] = first(rows, lambda r: abs(r["V_relative_error"]) > percent / 100)
    for state in ("mu", "V"):
        t = first(rows, lambda r: r["next_" + state] < 0)
        event["negative_" + state + "_from_step"] = t
        event["negative_" + state + "_to_step"] = t + 1 if t is not None else None
    return event


def window_summary(base, rows, kappa, start):
    result = []
    for name, rs in [("early", [r for r in rows if r["step"] < start + 5]),
                     ("late", [r for r in rows if r["step"] >= start + 5])]:
        if not rs:
            continue
        c = values(rs, "beta_measured") * values(rs, "V")
        q = values(rs, "Q_flux_residual") + kappa * values(rs, "beta_measured") * values(rs, "M3")
        row = {**base, "window": name, "points": len(rs), "step_min": rs[0]["step"], "step_max": rs[-1]["step"]}
        for field in ("C", "M3", "beta", "gamma", "delta_V", "Q_flux", "Q_moment", "Q_selection", "Q_total"):
            row[field + "_rmse"] = rms(values(rs, field + "_residual"))
        for field, target in [("C", c), ("M3", values(rs, "M3")), ("Q_total", q)]:
            row[field + "_relative_l2"] = row[field + "_rmse"] / rms(target) if rms(target) else None
        for field in ("beta", "gamma"):
            row[field + "_weight_share"] = sum(values(rs, field + "_fit_weight")) / sum(values(rows, field + "_fit_weight"))
        result.append(row)
    return result


def diagnose(parameters, metrics, trajectories, residuals):
    paths, residual_groups = grouped(trajectories), grouped(residuals)
    statuses = {key(r): r["status"] for r in metrics}
    points, events, windows = [], [], []
    for p in parameters:
        base = {k: p[k] for k in ("run_id", "trait_id", "model", "is_control")}
        path, rs = sorted(paths[key(p)], key=lambda r: r["step"]), sorted(residual_groups[key(p)], key=lambda r: r["step"])
        by_step = {r["step"]: r for r in rs}
        b, g = [p[k] for k in ("c0", "c1", "c2")], [p[k] for k in ("gamma0", "gamma1", "gamma2") if np.isfinite(p[k])]
        local = []
        for r in path:
            t, mu, v = r["step"], r["mu_generated"], r["V_generated"]
            row = {**base, **{k: r[k] for k in ("step", "mu_observed", "V_observed", "direct_mu", "mu_generated", "V_generated")}}
            old = by_step.get(t)
            for name, state in [("measured", (old["mu"], old["V"], old["M3"]) if old else (np.nan,) * 3),
                                ("fitted", (old["mu"], old["V"], old["M3_fitted"]) if old else (np.nan,) * 3)]:
                row[name + "_margin"], row[name + "_relative_margin"] = moment_margin(*state)
            bhat = polyval(mu, b)
            mhat = polyval(t if p["gamma_predictor"] == "step" else mu, g) * v ** 1.5
            row.update(beta_generated=bhat, M3_generated=mhat)
            row["generated_margin"], row["generated_relative_margin"] = moment_margin(mu, v, mhat)
            row["V_error"] = v - r["V_observed"]
            row["V_relative_error"] = row["V_error"] / r["V_observed"] if r["V_observed"] > 0 else np.nan
            row["next_mu"] = mu + bhat * v if t < path[-1]["step"] else np.nan
            row["next_V"] = v + p["kappa"] * bhat * mhat - (bhat * v) ** 2 if t < path[-1]["step"] else np.nan
            local.append(row)
        events.append(event_summary(base, statuses[key(p)], b, local))
        points.extend(local)
        windows.extend(window_summary(base, rs, p["kappa"], path[0]["step"]))
    return points, events, windows


def report(output, parameters, points, events, windows):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def save(fig, name, caption):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=125, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        figures.append(f"<figure><img src='data:image/png;base64,{encoded}' alt='{name}'><figcaption>{caption}</figcaption></figure>")

    selected = [r for r in events if r["model"] == PRIMARY]
    runs = sorted({r["run_id"] for r in selected})
    fig, axs = plt.subplots(1, len(runs), figsize=(4.8 * len(runs), 7), squeeze=False)
    for ax, rid in zip(axs[0], runs):
        rs = sorted([r for r in selected if r["run_id"] == rid], key=lambda r: int(r["trait_id"]))
        for field, marker, color, label in [("generated_first_violation_step", "D", "C1", "moment violation (state t)"),
                                            ("V_departure_25_step", "s", "C4", "variance departure >25% (state t)"),
                                            ("negative_mu_from_step", "x", "red", "negative mean update (from t)"),
                                            ("negative_V_from_step", "+", "black", "negative variance update (from t)")]:
            ax.scatter(values(rs, field), range(len(rs)), marker=marker, label=label, s=35, color=color,
                       facecolors="none" if marker == "s" else color)
        ax.set(yticks=range(len(rs)), yticklabels=[r["trait_id"] + (" C" if r["is_control"] == "True" else "") +
               (" · done" if r["status"] == "complete" else " · stopped") for r in rs], xlabel="checkpoint / update source step", title=rid)
        ax.invert_yaxis()
        ax.set_xlim(min(values(points, "step")) - .5, max(values(points, "step")) + .5)
        ax.grid(axis="x", alpha=.2)
    axs[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -.10))
    save(fig, "event_order", "Quadratic state shape law, every series; C marks controls. Missing markers mean no event was detected before the saved horizon or stopping point. Variance departure compares generated and observed panel variance at the same checkpoint; 10% and 50% timings remain in the CSV. Update markers refer to t, not t+1: M3 at t cannot cause the mean update from that same state. Event ordering alone does not identify causation.")

    ps, es = {key(r): r for r in parameters}, {key(r): r for r in events}
    groups = grouped(points)
    examples = [k for k in [("sae-rerun-s2", "25273", PRIMARY), ("sae-rerun-s3", "22348", PRIMARY),
                            ("sae-rerun-s3", "14034", PRIMARY)] if k in groups]
    if not examples:
        examples = [key(selected[0])]
    fig, axs = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5), squeeze=False)
    for ax, k in zip(axs[0], examples):
        rs, p = groups[k], ps[k]
        mus, vs = np.r_[values(rs, "mu_observed"), values(rs, "mu_generated")], np.r_[values(rs, "V_observed"), values(rs, "V_generated")]
        x = np.geomspace(min(mus[mus > 0]) * .7, max(mus[np.isfinite(mus)]) * 1.2, 200)
        y = np.geomspace(min(vs[vs > 0]) * .7, max(vs[np.isfinite(vs)]) * 1.2, 200)
        mu, v = np.meshgrid(x, y)
        b = polyval(mu, [p[c] for c in ("c0", "c1", "c2")])
        g = polyval(mu, [p[c] for c in ("gamma0", "gamma1", "gamma2")])
        bad = moment_margin(mu, v, g * v ** 1.5)[1] < -MOMENT_RTOL
        ax.contourf(x, y, bad, levels=[.5, 1.5], colors=["lightgrey"], alpha=.7)
        ax.contourf(x, y, mu + b * v < 0, levels=[.5, 1.5], colors=["salmon"], alpha=.5)
        ax.plot(values(rs, "mu_observed"), values(rs, "V_observed"), "ko-", ms=3, label="measured panels")
        ax.plot(values(rs, "mu_generated"), values(rs, "V_generated"), "o-", ms=3, color="C0", label="generated")
        for field, marker, color in [("generated_first_violation_step", "D", "C1"), ("negative_mu_from_step", "x", "red")]:
            hit = next((r for r in rs if r["step"] == es[k][field]), None)
            if hit:
                ax.scatter(hit["mu_generated"], hit["V_generated"], marker=marker, color=color, s=75, zorder=5)
                ax.annotate(f"t={int(hit['step'])}", (hit["mu_generated"], hit["V_generated"]), xytext=(5, 5), textcoords="offset points")
        for i, root in enumerate(es[k]["selection_roots"]):
            if x[0] <= root <= x[-1]:
                ax.axvline(root, color="C2", ls="--", label="beta(mu)=0" if i == 0 else "_nolegend_")
        ax.set(xscale="log", yscale="log", xlabel="mu", ylabel="V", title=f"{k[0]} · {k[1]}")
        ax.legend(fontsize=8)
    save(fig, "state_paths", "Examples, where present: display feature 25273 in s2; the earlier negative-mean case 22348 in s3 (now completed); and control 14034 in s3, illustrating a moment violation and an impossible mean update at the same source state. These panels illustrate specific cases; the event table covers all series. Grey tests fitted M3(mu,V): measured points within it indicate an inconsistent fitted third moment, not a violation by their measured moments. Red marks mu + beta(mu)V < 0. Diamonds mark the first moment violation and crosses the source state of a negative mean update. Log axes show positive states only; the invalid attempted next mean is not plotted as a valid state. Dashed lines are selection roots, not evidence of a stable equilibrium of the joint system.")

    display = examples[0][:2]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, field in zip(axs.flat, ("C_relative_l2", "M3_relative_l2", "Q_total_relative_l2", "beta_rmse", "gamma_rmse", "gamma_weight_share")):
        for label, (model, _, _) in zip(LABELS, SAE_SHAPE):
            rs = [r for r in windows if (r["run_id"], r["trait_id"]) == display and r["model"] == model]
            ax.plot([0 if r["window"] == "early" else 1 for r in rs], values(rs, field), "o-", label=label)
        ax.set(xticks=[0, 1], xticklabels=["first 5 steps", "remaining steps"], ylabel=field)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"{display[0]} · {display[1]}: fixed coefficients, early/late residuals")
    save(fig, "early_late", "The split is at initial step + 5, not after five valid rows. Relative L2 denominators use measured C, M3, or Q within each window. Raw beta/gamma RMSE and each window's share of the original fitting weights are separate diagnostics. C and beta curves coincide because selection stays fixed. Windows never refit coefficients; all run/feature results and signed-flux-component RMSEs are exported.")

    counts = []
    for label, (model, _, _) in zip(LABELS, SAE_SHAPE):
        rs = [r for r in events if r["model"] == model]
        counts.append(dict(model=label, series=len(rs), complete=sum(r["status"] == "complete" for r in rs),
                           measured_violation=sum(r["measured_violation_count"] > 0 for r in rs),
                           fitted_at_observed_violation=sum(r["fitted_violation_count"] > 0 for r in rs),
                           completed_with_generated_violation=sum(r["status"] == "complete" and r["generated_violation_count"] > 0 for r in rs),
                           stopped_with_generated_violation=sum(r["status"] != "complete" and r["generated_violation_count"] > 0 for r in rs)))
    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>SAE moment consistency</title>",
            "<style>body{max-width:1200px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{display:block;overflow:auto;border-collapse:collapse;font-size:.8rem}th,td{padding:.4rem;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>SAE moment consistency with fixed equations and coefficients</h1>",
            "<p>Read only the preceding shape-study exports. No coefficients, objectives, equations, or generated paths change. Four shape laws remain separate, with fixed per-series quadratic selection and kappa. These are development diagnostics, with dependent features within each run.</p>",
            "<p>For z ≥ 0, Cauchy–Schwarz gives E[z]E[z³] ≥ E[z²]², hence D = mu M3 + mu²V − V² ≥ 0. We export D and D/(|mu M3| + mu²V + V²), defining the all-zero case as zero. A relative margin below −10⁻¹² flags failure of this necessary condition; the tolerance addresses roundoff, not sampling uncertainty. Passing this condition does not prove that every support constraint or the transition law is valid.</p>",
            "<p>Measured and fitted-at-observed moments use the preceding common finite fitting rows. Generated moment checks include the terminal state when valid, but next-update checks stop at the requested horizon. Missing terminal panel moments and post-stop generated states stay missing. Independent direct means remain separate and never supply variance or sample-panel moments.</p>",
            "<p>Variance departure means |V_generated − V_observed|/V_observed > 25%, with 10% and 50% sensitivity timings exported. Only positive observed panel variances allow this ratio; errors are not interpolated or treated as calibrated uncertainty. Moment events occur at state t; impossible-update events retain both source t and destination t+1. M3 at t changes V at t+1, not mu at t+1. Event order can be consistent with a proposed mechanism without establishing causation.</p>",
            table(counts, list(counts[0]))]
    html.extend(figures)
    for rid in runs:
        rs = [r for r in selected if r["run_id"] == rid]
        html.extend([f"<details><summary>{escape(rid)}: quadratic-state event times</summary>",
                     table(rs, ["trait_id", "status", "fitted_first_violation_step", "generated_first_violation_step", "V_departure_10_step", "V_departure_25_step", "V_departure_50_step", "negative_mu_from_step", "negative_mu_to_step", "selection_roots"]), "</details>"])
    html.append("<p>Three CSVs retain per-checkpoint diagnostics, per-series events, and early/late errors for every law. Stop for discussion before changing a closure or fitting objective.</p></html>")
    (output / "index.html").write_text("\n".join(html))


def run(fits, output):
    fits, output = Path(fits), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    parameters, metrics, trajectories, residuals = [read_export(fits / f"{name}.csv") for name in ("parameters", "metrics", "trajectories", "residuals")]
    points, events, windows = diagnose(parameters, metrics, trajectories, residuals)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("points", points), ("events", events), ("windows", windows)]:
        write_csv(output / f"{name}.csv", [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                                          for k, v in r.items()} for r in rows])
    report(output, parameters, points, events, windows)
    print(f"Diagnosed {len(events)} saved paths; report: {output / 'index.html'}")
    return points, events, windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.fits, args.output)
