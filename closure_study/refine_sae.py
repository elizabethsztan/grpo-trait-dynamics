"""Per-run SAE selection curvature with fixed skewness and flux correction."""

import argparse
import base64
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyval

from .fit_controlled import table, values
from .fit_llm import SAE_SELECTION, fit_series
from .io import read_jsonl, write_csv
import matplotlib.pyplot as plt


LABELS = ["old beta OLS", "state affine", "state quadratic", "time affine", "time quadratic"]


def report(output, groups, parameters, metrics, trajectories, residuals, feature):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def subset(records, key, model=None):
        return [r for r in records if (r["run_id"], r["trait_id"]) == key and (model is None or r["model"] == model)]

    def save(fig, name, caption):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=125, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        figures.append(f"<figure><img src='data:image/png;base64,{encoded}' alt='{name}'><figcaption>{caption}</figcaption></figure>")

    keys = sorted(key for key in groups if key[1] == feature)
    key = next((k for k in keys if k[0] == "sae-rerun-s2"), keys[0])
    rows = groups[key]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, x in zip(axs[0], ("mu", "step")):
        ax.scatter(values(rows, x), values(rows, "beta"), color="black", s=15, label="measured beta")
        for i, (name, predictor, _) in enumerate(SAE_SELECTION):
            if predictor != x:
                continue
            p = subset(parameters, key, name)[0]
            grid = np.linspace(np.nanmin(values(rows, x)), np.nanmax(values(rows, x)), 150)
            ax.plot(grid, polyval(grid, [p[k] for k in ("c0", "c1", "c2") if p[k] is not None]), color=f"C{i}", label=LABELS[i])
        ax.set(xlabel=x, ylabel="beta")
        ax.legend(fontsize=8)
    axs[1, 0].plot(values(rows, "step"), values(rows, "C"), "ko", ms=3, label="measured C")
    for i, (name, _, _) in enumerate(SAE_SELECTION):
        rs = subset(residuals, key, name)
        axs[1, 0].plot(values(rs, "step"), values(rs, "beta_fitted") * values(rs, "V"), color=f"C{i}", label=LABELS[i])
    axs[1, 0].set(xlabel="step", ylabel="C at measured states")
    axs[1, 0].legend(fontsize=8)
    for k in keys:
        p = subset(parameters, k, "flux_state_quadratic")[0]
        mu = values(groups[k], "mu")
        grid = np.linspace(np.nanmin(mu), np.nanmax(mu), 150)
        axs[1, 1].plot(grid, polyval(grid, [p["c0"], p["c1"], p["c2"]]), label=k[0])
    axs[1, 1].set(xlabel="mu", ylabel="quadratic beta(mu), per run")
    axs[1, 1].legend(fontsize=8)
    fig.suptitle(f"Feature {feature}: {key[0]} (three panels); all runs (bottom right)")
    save(fig, "selection_laws", "The old affine beta fit separates changing the objective from adding curvature. All four flux fits minimize sum(C − V beta_hat)^2, equivalent to weighting squared beta residuals by V². This is not inverse-noise weighting. Curves are shown only over each run's measured range, and coefficients are not shared.")

    fig, axs = plt.subplots(2, len(keys), figsize=(4.4 * len(keys), 7), squeeze=False)
    for j, k in enumerate(keys):
        for ax, state in zip(axs[:, j], ("mu", "V")):
            for i, (name, _, _) in enumerate(SAE_SELECTION):
                rs = subset(trajectories, k, name)
                ax.plot(values(rs, "step"), values(rs, f"{state}_generated"), color=f"C{i}", label=LABELS[i])
            ax.plot(values(rs, "step"), values(rs, f"{state}_observed"), "ko", ms=3, label="sample panel")
            if state == "mu":
                ax.plot(values(rs, "step"), values(rs, "direct_mu"), "x", color="grey", label="separate direct mean")
            ax.set(xlabel="step", ylabel=state, title=k[0] if state == "mu" else None)
    axs[0, 0].legend(fontsize=7)
    save(fig, "generated_trajectories", "Only initial sample-panel mean/variance enter generation; time rivals also use the clock. Every law uses that run's unchanged affine gamma(mu) and kappa. Missing final panel moments remain missing; separate direct means are shown only where saved. These are whole-development-run reconstructions, not held-out predictions.")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, field, label in [(axs[0], "C_residual", "C residual"), (axs[1], "Q_selection_residual", "Q selection residual")]:
        for i, (name, _, _) in enumerate(SAE_SELECTION):
            rs = subset(residuals, key, name)
            ax.plot(values(rs, "step"), values(rs, field), color=f"C{i}", label=LABELS[i])
        ax.set(xlabel="step", ylabel=label)
    for field, label in [("Q_flux_residual", "Q − kappa beta M3"), ("Q_moment_residual", "kappa beta (M3 − fitted M3)")]:
        axs[2].plot(values(rs, "step"), values(rs, field), label=label)
    axs[2].set(xlabel="step", ylabel="Fixed Q residual components")
    for ax in axs:
        ax.axhline(0, color="grey", lw=.7)
    axs[0].legend(fontsize=7)
    axs[2].legend(fontsize=7)
    save(fig, "residual_components", f"Feature {escape(feature)}, run {escape(key[0])}. Q_total = Q_flux + Q_moment + Q_selection. Flux and moment components are identical across selection models; signed errors can cancel. Exports also retain beta residuals and the complete variance-increment residual, including the squared mean increment.")

    runs = sorted({k[0] for k in groups})
    features = sorted({k[1] for k in groups}, key=int)
    fig, axs = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 6), squeeze=False)
    for ax, rid in zip(axs[0], runs):
        grid = np.full((len(features), len(SAE_SELECTION)), np.nan)
        for r in metrics:
            if r["run_id"] == rid:
                grid[features.index(r["trait_id"]), [name for name, _, _ in SAE_SELECTION].index(r["model"])] = int(r["status"] == "complete")
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("lightgrey")
        ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        labels = [f + (" (control)" if next(r for r in parameters if r["trait_id"] == f).get("is_control") else "") for f in features]
        ax.set(xticks=range(len(SAE_SELECTION)), xticklabels=LABELS, yticks=range(len(features)), yticklabels=labels, title=rid)
        ax.tick_params(axis="x", rotation=60)
    save(fig, "completion", "Green means generation finished; red means it left the checked support or became nonfinite; grey means absent. Completion is not an adequacy test. Failed paths are retained without clipping and have no full-horizon RMSE; comparisons of errors must account for differing completion sets. Features within a run are dependent.")

    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>SAE selection curvature</title>",
            "<style>body{max-width:1150px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{display:block;overflow:auto;border-collapse:collapse;font-size:.8rem}th,td{padding:.4rem;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>SAE selection curvature with fixed skewness and kappa</h1>",
            "<p>Compare affine and quadratic beta(mu) with matched affine and quadratic beta(t). All four minimize squared raw covariance residuals C − V beta_hat using the same finite rows with positive variance. Keep the previous unweighted affine beta fit as a reference. Affine state/time laws have two coefficients each; quadratic laws have three. V² weights are exported for the covariance objective, and unit weights for the old beta objective. Finite-sample C need not be accurately measured; this fitting choice does not calibrate uncertainty.</p>",
            "<p>Fit gamma(mu) and kappa once by the previous per-run procedure and hold them unchanged across selection candidates. There is no pooling. Generate mu' = mu + beta_hat V and V' = V + kappa beta_hat gamma_hat(mu) V^(3/2) − (beta_hat V)². Only selection uses time in a time rival; the skewness law continues to use generated mu.</p>",
            "<p>The variance-increment residual compares Q − C² with kappa beta_hat M3_hat − (beta_hat V)² at measured states. It is derived from measured fluxes, not from a direct next-variance measurement.</p>",
            "<p>Nonfinite states, negative mean, or negative variance stop generation. Mean and variance scores use old-policy panels; independent direct mean levels have a separate score where saved. Scores exclude the supplied initial state. Missing final panel moments remain missing. Whole-run development fits do not establish prospective validity.</p>",
            f"<p>Display feature {escape(feature)} is explicitly chosen (default 25273, the first selected feature in the original pool), rather than selected by fit quality. Four figures and four CSVs retain per-run coefficients, metrics, trajectories, and residuals. Inspect both covariance and beta residuals: a smaller C residual can coexist with a larger beta residual at small variance.</p>"]
    html.extend(figures)
    for k in sorted(groups):
        html.extend([f"<details><summary>{escape(k[0])} · feature {escape(k[1])}</summary>",
                     table(subset(parameters, k), ["model", "selection_objective", "c0", "c1", "c2", "gamma0", "gamma1", "kappa", "fit_points"]),
                     table(subset(metrics, k), ["model", "status", "mu_rmse", "V_rmse", "direct_mu_rmse", "C_relative_l2", "beta_rmse"] +
                           [f"Q_{part}_residual_relative_l2" for part in ("flux", "moment", "selection", "total")]), "</details>"])
    html.append("<p>Stop for discussion before sharing coefficients, changing skewness/kappa, or planning confirmation training.</p></html>")
    (output / "index.html").write_text("\n".join(html))


def run(table_path, output, feature="25273"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    groups = defaultdict(list)
    for r in read_jsonl(table_path):
        if r["family"] == "llm_continuous":
            groups[(r["run_id"], r["trait_id"])].append(r)
    if feature not in {key[1] for key in groups}:
        raise ValueError("supply SAE data including the requested display feature")
    combined = [[], [], [], []]
    for _, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["step"])
        for target, records in zip(combined, fit_series(rows, sae_selection=True)):
            target.extend(records)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in zip(("parameters", "metrics", "trajectories", "residuals"), combined):
        write_csv(output / f"{name}.csv", [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                                          for k, v in r.items()} for r in records])
    report(output, groups, *combined, feature)
    print(f"Compared {len(groups)} SAE run/feature series, {len(combined[1])} paths; report: {output / 'index.html'}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="25273")
    args = parser.parse_args()
    run(args.table, args.output, args.feature)
