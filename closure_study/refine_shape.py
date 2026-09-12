"""Third-moment shape fits and local failure diagnostics with fixed selection."""

import argparse
import base64
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyval

from .fit_controlled import selection_value, table, values
from .fit_llm import SAE_SHAPE, fit_series
from .io import read_jsonl, write_csv
import matplotlib.pyplot as plt


LABELS = ["old gamma OLS", "M3 state affine", "M3 state quadratic", "M3 time quadratic"]
NEXT_FIELDS = ["V_next_fitted", "V_next_measured_M3", "V_next_measured_beta", "V_next_measured_flux",
               "mu_next_fitted", "mu_next_measured_flux"]


def subset(records, key, model=None):
    return [r for r in records if (r["run_id"], r["trait_id"]) == key and (model is None or r["model"] == model)]


def failure_audit(parameters, metrics, trajectories, residuals):
    audit = []
    for m in metrics:
        key, model = (m["run_id"], m["trait_id"]), m["model"]
        rs = subset(residuals, key, model)
        record = {k: m[k] for k in ("run_id", "trait_id", "model", "status")}
        record["observed_update_count"] = len(rs)
        for field in NEXT_FIELDS:
            bad = [r["step"] for r in rs if r[field] < 0]
            record[field + "_negative_count"] = len(bad)
            record[field + "_first_negative_step"] = min(bad) if bad else None
        if m["status"] != "complete":
            path = [r for r in subset(trajectories, key, model) if np.isfinite(r["mu_generated"])]
            last, p = path[-1], subset(parameters, key, model)[0]
            mu, v, step = last["mu_generated"], last["V_generated"], last["step"]
            record.update(last_valid_step=step, last_mu=mu, last_V=v)
            for state in ("mu", "V"):
                lo, hi = min(values(rs, state)), max(values(rs, state))
                record[state + "_fit_min"], record[state + "_fit_max"] = lo, hi
                outside = [r["step"] for r in path if not lo <= r[state + "_generated"] <= hi]
                record[state + "_first_range_exit_step"] = outside[0] if outside else None
                record[state + "_last_state_outside_range"] = not lo <= last[state + "_generated"] <= hi
            with np.errstate(over="ignore", invalid="ignore"):
                b = selection_value([p["c0"], p["c1"], p["c2"]], mu, v, step, p.get("predictor", "mu"))
                g = polyval(step if p["gamma_predictor"] == "step" else mu,
                            [p[k] for k in ("gamma0", "gamma1", "gamma2") if p[k] is not None])
                record["attempted_next_mu"] = mu + b * v
                record["attempted_next_V"] = v + p["kappa"] * b * g * v ** 1.5 - (b * v) ** 2
        audit.append(record)
    return audit


def report(output, groups, parameters, metrics, trajectories, residuals, audit, feature):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

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
        ax.scatter(values(rows, x), values(rows, "skewness"), color="black", s=15, label="measured gamma")
        for i, (name, predictor, _) in enumerate(SAE_SHAPE):
            if predictor != x:
                continue
            p = subset(parameters, key, name)[0]
            grid = np.linspace(np.nanmin(values(rows, x)), np.nanmax(values(rows, x)), 150)
            ax.plot(grid, polyval(grid, [p[k] for k in ("gamma0", "gamma1", "gamma2") if p[k] is not None]), color=f"C{i}", label=LABELS[i])
        ax.set(xlabel=x, ylabel="gamma")
        ax.legend(fontsize=8)
    axs[1, 0].plot(values(rows, "step"), values(rows, "M3"), "ko", ms=3, label="measured M3")
    for i, (name, _, _) in enumerate(SAE_SHAPE):
        rs = subset(residuals, key, name)
        axs[1, 0].plot(values(rs, "step"), values(rs, "M3_fitted"), color=f"C{i}", label=LABELS[i])
    axs[1, 0].set(xlabel="step", ylabel="M3 at measured states")
    axs[1, 0].legend(fontsize=8)
    for k in keys:
        p = subset(parameters, k, "moment_state_quadratic")[0]
        mu = values(groups[k], "mu")
        grid = np.linspace(np.nanmin(mu), np.nanmax(mu), 150)
        axs[1, 1].plot(grid, polyval(grid, [p["gamma0"], p["gamma1"], p["gamma2"]]), label=k[0])
    axs[1, 1].set(xlabel="mu", ylabel="quadratic gamma(mu), per run")
    axs[1, 1].legend(fontsize=8)
    fig.suptitle(f"Feature {feature}: {key[0]} (three panels); all runs (bottom right)")
    save(fig, "shape_laws", "M3 fits minimize sum(M3 − V^(3/2) gamma_hat)^2, equivalent to V³-weighted squared gamma residuals. The old unweighted affine-gamma fit remains a reference. Weighting can improve M3 while worsening raw gamma, and does not calibrate noise. Curves use each run's measured range; coefficients remain separate.")

    fig, axs = plt.subplots(2, len(keys), figsize=(4.4 * len(keys), 7), squeeze=False)
    for j, k in enumerate(keys):
        for ax, state in zip(axs[:, j], ("mu", "V")):
            for i, (name, _, _) in enumerate(SAE_SHAPE):
                rs = subset(trajectories, k, name)
                ax.plot(values(rs, "step"), values(rs, f"{state}_generated"), color=f"C{i}", label=LABELS[i])
            ax.plot(values(rs, "step"), values(rs, f"{state}_observed"), "ko", ms=3, label="sample panel")
            if state == "mu":
                ax.plot(values(rs, "step"), values(rs, "direct_mu"), "x", color="grey", label="separate direct mean")
            ax.set(xlabel="step", ylabel=state, title=k[0] if state == "mu" else None)
    axs[0, 0].legend(fontsize=7)
    save(fig, "generated_trajectories", "Each trajectory starts from its measured initial mean/variance. Quadratic beta(mu) and kappa are unchanged across shape models. Only gamma uses the clock in the time rival. Later means and variances are generated, never reset. Final panel moments stay missing; direct mean levels are separate where saved.")

    baseline_failures = [r for r in audit if r["model"] == "gamma_ols_affine" and r["status"].startswith("negative_variance")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fields = NEXT_FIELDS[:4]
    axs[0].bar(range(4), [sum(r[f + "_negative_count"] > 0 for r in baseline_failures) for f in fields])
    axs[0].set(xticks=range(4), xticklabels=["fitted beta/M3", "measured M3", "measured beta", "measured Q/C"],
               ylabel="Series with a negative observed-state V update", title=f"Baseline variance-failure cohort: {len(baseline_failures)} series")
    axs[0].tick_params(axis="x", rotation=25)
    bottom = np.zeros(len(SAE_SHAPE))
    for category, label in [("complete", "complete"), ("negative_variance", "negative variance"),
                             ("mean_outside_support", "negative mean"), ("nonfinite_state", "nonfinite")]:
        count = np.array([sum(r["model"] == name and r["status"].startswith(category) for r in metrics) for name, _, _ in SAE_SHAPE])
        axs[1].bar(range(len(SAE_SHAPE)), count, bottom=bottom, label=label)
        bottom += count
    axs[1].set(xticks=range(len(SAE_SHAPE)), xticklabels=LABELS, ylabel="Run/feature series", title="Generated trajectories: all series")
    axs[1].tick_params(axis="x", rotation=25)
    axs[1].legend(fontsize=8)
    save(fig, "failure_diagnostics", "Left: one-step substitutions at observed states within the baseline negative-variance cohort; measured-M3 and measured-beta substitutions retain kappa. Measured Q/C uses V + Q − C². These are local diagnostics, not generated trajectories or independent next-variance measurements. Right: first recorded stopping reason; an attempted update can violate both mean and variance bounds.")

    runs = sorted({k[0] for k in groups})
    features = sorted({k[1] for k in groups}, key=int)
    fig, axs = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 6), squeeze=False)
    for ax, rid in zip(axs[0], runs):
        grid = np.full((len(features), len(SAE_SHAPE)), np.nan)
        for r in metrics:
            if r["run_id"] == rid:
                grid[features.index(r["trait_id"]), [name for name, _, _ in SAE_SHAPE].index(r["model"])] = int(r["status"] == "complete")
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("lightgrey")
        ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        labels = [f + (" (control)" if next(r for r in parameters if r["trait_id"] == f).get("is_control") else "") for f in features]
        ax.set(xticks=range(len(SAE_SHAPE)), xticklabels=LABELS, yticks=range(len(features)), yticklabels=labels, title=rid)
        ax.tick_params(axis="x", rotation=60)
    save(fig, "completion", "Green means completed; red means stopped; grey means absent. No states are clipped and failed paths have no full-horizon scores. Completion does not establish adequacy. Features within each run are dependent, and error comparisons must account for differing completion sets.")

    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>SAE third-moment closure</title>",
            "<style>body{max-width:1150px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{display:block;overflow:auto;border-collapse:collapse;font-size:.8rem}th,td{padding:.4rem;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>SAE third-moment closure with fixed selection and kappa</h1>",
            "<p>Keep each run/feature's quadratic beta(mu), fitted through C = V beta, and its measured-Q kappa unchanged. Compare the old affine gamma(mu) fit with affine/quadratic gamma(mu) and a matched quadratic gamma(t), all three new laws fitted through M3 = V^(3/2) gamma. Coefficients, residuals, and fitting weights use the same finite rows with positive variance. No coefficients are pooled.</p>",
            "<p>Generate mu' = mu + beta_hat(mu) V and V' = V + kappa beta_hat(mu) gamma_hat V^(3/2) − (beta_hat(mu) V)². Only gamma uses time in the time rival. All models receive only initial panel moments after fitting; missing final panel moments stay missing and independent direct means have a separate score where saved. Scores exclude the initial state. These whole-run development fits do not establish prospective validity.</p>",
            "<p>Q residuals retain the exact decomposition (Q − kappa beta M3) + kappa beta(M3 − fitted M3) + kappa(beta − fitted beta) fitted M3. The covariance and Q-flux residuals stay fixed. Both remaining Q terms can change when fitted M3 changes, despite fixed selection coefficients, and signed errors can cancel. The variance-increment residual compares Q − C² with the model increment at measured states.</p>",
            "<p>The failure audit records negative observed-state updates and reconstructs attempted updates after the last valid generated state. It also records exits from marginal fitted-mu and fitted-V ranges. Those ranges are descriptive: being inside them does not establish joint state coverage, and leaving them does not establish a cause of failure. Substitutions using measured beta, M3, or Q/C are diagnostics only and never enter generation.</p>",
            f"<p>Display feature {escape(feature)} is chosen explicitly (default 25273, the first selected feature in the original pool), rather than by fit quality. Four figures and five CSVs retain parameters, metrics, trajectories, residuals, and the failure audit. Raw gamma residuals and unit/V³ fitting weights remain available; sampling uncertainty is not calibrated.</p>"]
    html.extend(figures)
    failures = [r for r in audit if r["model"] == "gamma_ols_affine" and r["status"] != "complete"]
    html.extend(["<details><summary>Baseline failures: observed and generated states</summary>",
                 table(failures, ["run_id", "trait_id", "status"] + [f + "_negative_count" for f in NEXT_FIELDS] +
                       ["last_valid_step", "last_mu", "last_V", "mu_last_state_outside_range", "V_last_state_outside_range", "attempted_next_mu", "attempted_next_V"]), "</details>"])
    for k in sorted(groups):
        html.extend([f"<details><summary>{escape(k[0])} · feature {escape(k[1])}</summary>",
                     table(subset(parameters, k), ["model", "shape_objective", "c0", "c1", "c2", "gamma0", "gamma1", "gamma2", "kappa", "fit_points"]),
                     table(subset(metrics, k), ["model", "status", "mu_rmse", "V_rmse", "direct_mu_rmse", "M3_relative_l2", "gamma_rmse"] +
                           [f"Q_{part}_residual_relative_l2" for part in ("flux", "moment", "selection", "total")]), "</details>"])
    html.append("<p>Stop for discussion before sharing coefficients, expanding the closure family, or planning confirmation training.</p></html>")
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
        for target, records in zip(combined, fit_series(rows, sae_shape=True)):
            target.extend(records)
    audit = failure_audit(*combined)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in zip(("parameters", "metrics", "trajectories", "residuals", "failure_audit"), [*combined, audit]):
        write_csv(output / f"{name}.csv", [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                                          for k, v in r.items()} for r in records])
    report(output, groups, *combined, audit, feature)
    print(f"Compared {len(groups)} SAE series, {len(combined[1])} paths; report: {output / 'index.html'}")
    return *combined, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature", default="25273")
    args = parser.parse_args()
    run(args.table, args.output, args.feature)
