"""Binary development transfer and SAE state/time residual diagnostics."""

import argparse
import base64
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyval

from .fit_controlled import rms, rollout, table, values
from .fit_llm import fit_binary_flux, fit_series, observations, transition_steps
from .io import read_jsonl, write_csv
from .transfer_controlled import other_runs
import matplotlib.pyplot as plt


BINARY = [("constant", "T", 0), ("state_affine", "T", 1), ("time_affine", "step", 1)]


def binary_transfer(runs):
    if len(runs) < 2 or len({(r["setting_id"], r["trait_id"]) for rows in runs.values() for r in rows}) != 1:
        raise ValueError("binary transfer requires multiple runs of one setting and trait")
    parameters, metrics, trajectories = [], [], []
    for rid, rows in sorted(runs.items()):
        step = transition_steps(rows)
        observed = observations(rows, True)[:, 0]
        if not np.isfinite(observed[0]) or not 0 <= observed[0] <= 1:
            raise ValueError("binary transfer requires a measured initial prevalence")
        for scope, training in [("own", rows), ("loo", other_runs(runs, rid))]:
            for name, predictor, degree in BINARY:
                coef, fit_mask, informative = fit_binary_flux(training, predictor, degree)
                base = {"run_id": rid, "setting_id": rows[0]["setting_id"], "trait_id": rows[0]["trait_id"],
                        "model": name, "scope": scope}
                parameters.append({**base, "training_runs": sorted({r["run_id"] for r in training}),
                                   "c0": coef[0], "c1": coef[1] if degree else None,
                                   "fit_points": int(fit_mask.sum()), "informative_points": informative})
                path, status = rollout([observed[0]], step, coef, time=predictor == "step")
                t, c, x = (values(rows, key) for key in ("T", "C", predictor))
                valid = np.isfinite(t) & np.isfinite(c) & np.isfinite(x)
                error = c[valid] - t[valid] * (1 - t[valid]) * polyval(x[valid], coef)
                metrics.append({**base, "status": status, "completed_updates": int(np.isfinite(path[1:, 0]).sum()),
                                "mu_rmse": rms(path[1:, 0] - observed[1:]) if status == "complete" else None,
                                "mu_score_points": int(np.isfinite(observed[1:]).sum()),
                                "C_relative_l2": rms(error) / rms(c[valid]) if rms(c[valid]) else None,
                                "C_score_points": int(valid.sum())})
                trajectories.extend({**base, "step": int(s), "mu_observed": observed[i], "mu_generated": path[i, 0]}
                                    for i, s in enumerate(np.r_[step, step[-1] + 1]))
    return parameters, metrics, trajectories


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    return float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.std(x) > 0 and np.std(y) > 0 else None


def sae_diagnostics(groups):
    points, summaries = [], []
    for _, rows in sorted(groups.items()):
        parameters, metrics, _, residuals = fit_series(rows)
        p = next(r for r in parameters if r["model"] == "affine_corrected")
        m = next(r for r in metrics if r["model"] == "affine_corrected")
        errors = {r["step"]: r for r in residuals if r["model"] == "affine_corrected"}
        series = []
        for row in rows:
            point = {key: row.get(key) for key in ("run_id", "setting_id", "trait_id", "is_control", "step",
                                                  "mu", "V", "beta", "C", "skewness", "M3", "Q")}
            if row["step"] in errors:
                point.update({key: value for key, value in errors[row["step"]].items() if key.endswith("_residual")})
                point["beta_residual"] = row["beta"] - polyval(row["mu"], [p["c0"], p["c1"]])
                point["gamma_residual"] = row["skewness"] - polyval(row["mu"], [p["gamma0"], p["gamma1"]])
            series.append(point)
        summary = {**p, **{key: value for key, value in m.items() if key.endswith("relative_l2")}}
        for name, x, y in [("beta_residual_V", "beta_residual", "V"), ("C_residual_V", "C_residual", "V"),
                           ("gamma_residual_time", "gamma_residual", "step")]:
            summary["corr_" + name] = correlation(values(series, x), values(series, y))
        points.extend(series)
        summaries.append(summary)
    return points, summaries


def report(output, parameters, metrics, trajectories, points, summaries, feature):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def save(fig, name, caption):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=125, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        figures.append(f"<figure><img src='data:image/png;base64,{encoded}' alt='{name}'><figcaption>{caption}</figcaption></figure>")

    runs = sorted({r["run_id"] for r in trajectories})
    fig, axs = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.6), squeeze=False)
    for ax, rid in zip(axs[0], runs):
        for name, _, _ in BINARY:
            rows = [r for r in trajectories if r["run_id"] == rid and r["model"] == name and r["scope"] == "loo"]
            ax.plot(values(rows, "step"), values(rows, "mu_generated"), label=name)
        ax.plot(values(rows, "step"), values(rows, "mu_observed"), "ko", ms=3, label="observed")
        ax.set(xlabel="step", ylabel="T", title=rid.rsplit("seed", 1)[-1])
        ax.legend(fontsize=8)
    save(fig, "binary_transfer", "Each panel omits the target run when fitting. Curves use its initial T and the fitted law; the time rival also uses the clock. Scores use observed future checkpoints only. These are development transfer checks on inspected runs.")

    selected = [r for r in points if r["trait_id"] == feature]
    if not selected:
        raise ValueError(f"requested display feature {feature} is missing")
    sae_runs = sorted({r["run_id"] for r in selected})

    def panels(specs, name, caption, residual=False):
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (x, y, xlabel, ylabel) in zip(axs.flat, specs):
            for i, rid in enumerate(sae_runs):
                rows = [r for r in selected if r["run_id"] == rid]
                ax.plot(values(rows, x), values(rows, y), ".-", color=f"C{i}", ms=5, lw=.8, label=rid)
                if x == "mu" and y == "beta":
                    ax.scatter([rows[0][x]], [rows[0][y]], marker="s", color=f"C{i}", s=45)
                    ax.scatter([rows[-1][x]], [rows[-1][y]], marker="x", color=f"C{i}", s=45)
                    p = next(r for r in summaries if r["run_id"] == rid and r["trait_id"] == feature)
                    grid = np.linspace(np.nanmin(values(rows, x)), np.nanmax(values(rows, x)), 100)
                    ax.plot(grid, polyval(grid, [p["c0"], p["c1"]]), "--", color=f"C{i}", lw=.8)
            if residual:
                ax.axhline(0, color="grey", lw=.7)
            ax.set(xlabel=xlabel, ylabel=ylabel)
        axs[0, 1].legend(fontsize=8)
        fig.suptitle(f"SAE feature {feature}")
        save(fig, name, caption)

    panels([("mu", "beta", "mu", "beta"), ("step", "beta", "step", "beta"),
            ("step", "C", "step", "C"), ("step", "V", "step", "V")], "sae_selection",
           "The same feature is overlaid across runs. Solid lines connect measurements in time order; squares mark the initial state and crosses the last state in the state-space panel. Dashed lines show the existing per-run affine beta(mu) fits. Explicit beta-versus-time, raw C, and V panels distinguish the temporal shape from amplification by small variance. No SAE time law is fitted.")
    panels([("V", "beta_residual", "V", "beta - fitted beta(mu)"),
            ("V", "C_residual", "V", "C - fitted beta(mu) V"),
            ("mu", "gamma_residual", "mu", "gamma - fitted gamma(mu)"),
            ("step", "gamma_residual", "step", "gamma - fitted gamma(mu)")], "sae_residuals",
           "Selection and skewness residuals remain separate. The raw-C residual accompanies the beta residual to expose the effect of division by V. Lines connect measurements in time order. Apparent structure is descriptive: V, mu, and time covary, and correlations alone cannot identify a missing state variable.", residual=True)

    features = sorted({r["trait_id"] for r in summaries}, key=int)
    all_runs = sorted({r["run_id"] for r in summaries})
    fig, axs = plt.subplots(1, 3, figsize=(13, 6))
    names = [("corr_beta_residual_V", "corr(beta residual, V)"),
             ("corr_C_residual_V", "corr(C residual, V)"),
             ("corr_gamma_residual_time", "corr(gamma residual, time)")]
    for ax, (field, title) in zip(axs, names):
        grid = np.full((len(features), len(all_runs)), np.nan)
        for r in summaries:
            grid[features.index(r["trait_id"]), all_runs.index(r["run_id"])] = r[field] if r[field] is not None else np.nan
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("lightgrey")
        im = ax.imshow(grid, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        labels = [f + (" (control)" if next(r for r in summaries if r["trait_id"] == f).get("is_control") else "") for f in features]
        ax.set(xticks=range(len(all_runs)), xticklabels=all_runs, yticks=range(len(features)), yticklabels=labels, title=title)
        ax.tick_params(axis="x", rotation=60)
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "sae_feature_comparison", "Pearson correlations summarize each feature/run separately; grey means undefined. They use finite residual pairs from the common fit mask. There are only 20 serial observations per series in this archive, no sampling confidence intervals, and features within a run are dependent. Small correlation does not rule out nonlinear structure.")

    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>LLM transfer and selection diagnostics</title>",
            "<style>body{max-width:1150px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{display:block;overflow:auto;border-collapse:collapse;font-size:.8rem}th,td{padding:.4rem;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>LLM transfer and selection diagnostics</h1>",
            "<p>Binary: compare constant S, affine S(T), and affine S(t) in the above-hook setting. Fit C = T(1−T)S by unweighted least squares on measured old-state/flux pairs. Each leave-one-run-out fit pools the other whole runs within the same setting and trait; the target contributes only its initial T to generation. Own-run fits remain a reference. Affine state/time laws have two coefficients each; the constant has one. Invalid states stop without clipping or a full-horizon score.</p>",
            "<p>SAE: reuse the existing per-run affine beta(mu), affine gamma(mu), and kappa fits. No new continuous closure is fitted. All component residuals use the common finite, positive-variance mask. beta = C/V and gamma = M3/V^(3/2); ratio residuals can be amplified at small V. Signed Q flux, moment, and selection residuals remain available in the point table, and their relative L2 norms below use the same measured-Q denominator.</p>",
            f"<p>The display feature is {escape(feature)} (default 25273, the first selected feature in the original pool), chosen explicitly rather than by fit quality. All feature/run summaries remain available. State/time comparisons are descriptive; sampled estimates lack calibrated uncertainty and the existing trajectories may not distinguish the competing explanations.</p>",
            "<p>All runs are previously inspected development data. These transfer checks and residual plots do not provide prospective confirmation. Five CSVs retain binary parameters, metrics, trajectories, SAE points, and SAE summaries.</p>",
            "<h2>Binary transfer scores</h2>",
            table([r for r in metrics if r["scope"] == "loo"], ["run_id", "model", "status", "mu_rmse", "C_relative_l2", "mu_score_points", "C_score_points"])]
    html.extend(figures)
    html.extend(["<details><summary>Binary own-run and transferred coefficients / scores</summary>",
                 table(parameters, ["run_id", "scope", "model", "training_runs", "c0", "c1", "fit_points", "informative_points"]),
                 table(metrics, ["run_id", "scope", "model", "status", "mu_rmse", "C_relative_l2"]), "</details>"])
    fields = ["run_id", "c0", "c1", "gamma0", "gamma1", "kappa", "fit_points", "C_relative_l2"]
    fields += [f"Q_{part}_residual_relative_l2" for part in ("linear", "flux", "moment", "selection", "total")]
    fields += [name for name, _ in names]
    for feature_id in features:
        html.extend([f"<details><summary>SAE feature {escape(feature_id)}</summary>",
                     table([r for r in summaries if r["trait_id"] == feature_id], fields), "</details>"])
    html.append("<p>Stop for discussion before adding closure flexibility or planning new training.</p></html>")
    (output / "index.html").write_text("\n".join(html))


def run(table_path, output, feature="25273"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    binary, sae = defaultdict(list), defaultdict(list)
    for r in read_jsonl(table_path):
        if r["family"] == "llm_binary" and "above_hook" in r["setting_id"]:
            binary[r["run_id"]].append(r)
        if r["family"] == "llm_continuous":
            sae[(r["run_id"], r["trait_id"])].append(r)
    if not sae or feature not in {key[1] for key in sae}:
        raise ValueError("supply SAE data including the requested display feature")
    for rows in list(binary.values()) + list(sae.values()):
        rows.sort(key=lambda r: r["step"])
    parameters, metrics, trajectories = binary_transfer(binary)
    points, summaries = sae_diagnostics(sae)
    output.mkdir(parents=True, exist_ok=True)
    combined = [parameters, metrics, trajectories, points, summaries]
    for name, records in zip(("binary_parameters", "binary_metrics", "binary_trajectories", "sae_points", "sae_summary"), combined):
        write_csv(output / f"{name}.csv", [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                                          for k, v in r.items()} for r in records])
    report(output, *combined, feature)
    print(f"Compared {len(binary)} binary runs and diagnosed {len(sae)} SAE series; report: {output / 'index.html'}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="25273")
    args = parser.parse_args()
    run(args.table, args.output, args.feature)
