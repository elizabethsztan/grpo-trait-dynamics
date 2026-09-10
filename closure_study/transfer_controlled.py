"""Binary run transfer and one-parameter continuous flux correction; development only."""

import argparse
import base64
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyroots

from .fit_controlled import fit_curve, rms, rollout, table, validate_run, values
from .io import read_jsonl, write_csv
from .statistics import moments, probabilities


WINDOWS = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000)]
EARLY_STEPS = (0, 10, 25, 50, 100, 200)


def fit_kappa(rows):
    x, q = values(rows, "beta") * values(rows, "M3"), values(rows, "Q")
    valid = np.isfinite(x) & np.isfinite(q)
    x, q = x[valid], q[valid]
    if not x.size or x @ x == 0:
        raise ValueError("kappa is unidentified without nonzero measured beta*M3")
    return float(x @ q / (x @ x))


def other_runs(runs, omitted):
    return [r for rid, rows in runs.items() if rid != omitted for r in rows]


def upper_root(coef):
    roots = [float(r.real) for r in polyroots(coef) if abs(r.imag) < 1e-10 and 0 < r.real < 1]
    return max(roots) if roots else None


def simulate(rows, model, selection, skewness=None, *, time=False, kappa=1.):
    steps, mu, v = validate_run(rows)
    initial = [mu[0]] if skewness is None else [mu[0], v[0]]
    path, status = rollout(initial, steps.astype(int), selection, skewness, time, kappa=kappa)
    obs_mu, obs_v = np.r_[mu, rows[-1]["mu_next"]], np.r_[v, rows[-1].get("V_next", np.nan)]
    base = {"run_id": rows[0]["run_id"], "family": rows[0]["family"], "model": model}
    metric = {**base, "status": status, "completed_updates": int(np.isfinite(path[1:, 0]).sum()),
              "mu_rmse": rms(path[1:, 0] - obs_mu[1:]) if status == "complete" else None,
              "V_rmse": rms(path[1:, 1] - obs_v[1:]) if skewness is not None and status == "complete" else None}
    trajectories = [{**base, "step": int(step), "mu_observed": obs_mu[i], "mu_generated": path[i, 0],
                     "V_observed": obs_v[i] if skewness is not None else None,
                     "V_generated": path[i, 1] if skewness is not None else None}
                    for i, step in enumerate(np.r_[steps, steps[-1] + 1])]
    return metric, trajectories


def binary_transfer(runs):
    parameters, metrics, trajectories = [], [], []
    for rid, rows in sorted(runs.items()):
        for scope, training in [("own", rows), ("loo", other_runs(runs, rid))]:
            for law, predictor in [("state", "T"), ("time", "step")]:
                coef = fit_curve(values(training, predictor), values(training, "S"), 2)
                model = f"{law}_{scope}"
                parameters.append({"family": "tabular_binary", "run_id": rid, "model": model,
                                   "training_runs": sorted({r["run_id"] for r in training}),
                                   "c0": coef[0], "c1": coef[1], "c2": coef[2],
                                   "upper_root": upper_root(coef) if law == "state" else None})
                metric, path = simulate(rows, model, coef, time=law == "time")
                metrics.append(metric)
                trajectories.extend(path)
    return parameters, metrics, trajectories


def kappa_stability(runs):
    records = []
    for rid, rows in sorted(runs.items()):
        measured = [r for r in rows if r.get("Q") is not None]
        full = fit_kappa(measured)
        energy = sum((r["beta"] * r["M3"]) ** 2 for r in measured)
        for label, subset in [("whole_run", measured)] + [
                (f"{lo}:{hi}", [r for r in measured if lo <= r["step"] < hi]) for lo, hi in WINDOWS]:
            if not subset:
                continue
            x, q = values(subset, "beta") * values(subset, "M3"), values(subset, "Q")
            records.append({"run_id": rid, "window": label, "n": len(subset),
                            "kappa": fit_kappa(subset) if x @ x > 0 else None,
                            "fit_weight_share": float(x @ x / energy),
                            "correlation": float(np.corrcoef(x, q)[0, 1]) if np.std(x) > 0 and np.std(q) > 0 else None,
                            "relative_Q_error_full_kappa": rms(q - full * x) / rms(q) if rms(q) else None})
    return records


def continuous_transfer(runs):
    parameters, metrics, trajectories, residuals = [], [], [], []
    shared = fit_kappa([r for rows in runs.values() for r in rows])
    for rid, rows in sorted(runs.items()):
        training = other_runs(runs, rid)
        own_b = fit_curve(values(rows, "mu"), values(rows, "beta"), 1)
        own_g = fit_curve(values(rows, "mu"), values(rows, "skewness"), 1)
        loo_b = fit_curve(values(training, "mu"), values(training, "beta"), 1)
        loo_g = fit_curve(values(training, "mu"), values(training, "skewness"), 1)
        own_k, loo_k = fit_kappa(rows), fit_kappa(training)
        protocols = [("uncorrected_own", own_b, own_g, 1.), ("corrected_own", own_b, own_g, own_k),
                     ("shared_kappa", own_b, own_g, shared), ("loo_kappa", own_b, own_g, loo_k),
                     ("loo_all", loo_b, loo_g, loo_k)]
        for model, b, g, k in protocols:
            parameters.append({"family": "neural_continuous", "run_id": rid, "model": model,
                               "beta0": b[0], "beta1": b[1], "gamma0": g[0], "gamma1": g[1], "kappa": k,
                               "beta_gamma_training_runs": sorted(set(runs) - {rid}) if model == "loo_all" else [rid],
                               "kappa_training_runs": [] if model == "uncorrected_own" else sorted(runs) if model == "shared_kappa" else sorted(set(runs) - {rid}) if model.startswith("loo") else [rid]})
            metric, path = simulate(rows, model, b, g, kappa=k)
            valid = [r for r in rows if r.get("Q") is not None]
            mu, v, beta, m3, q = (values(valid, key) for key in ("mu", "V", "beta", "M3", "Q"))
            mhat, bhat = polyval(mu, g) * v ** 1.5, polyval(mu, b)
            components = {"Q_linear_residual": q - beta * m3, "Q_flux_residual": q - k * beta * m3,
                          "Q_moment_residual": k * beta * (m3 - mhat),
                          "Q_selection_residual": k * (beta - bhat) * mhat,
                          "Q_total_residual": q - k * bhat * mhat}
            for key, array in components.items():
                metric[key + "_relative_l2"] = rms(array) / rms(q) if rms(q) else None
                metric[key + "_sum"] = float(array.sum())
            residuals.extend({"run_id": rid, "model": model, "step": r["step"],
                              **{key: array[i] for key, array in components.items()}} for i, r in enumerate(valid))
            metrics.append(metric)
            trajectories.extend(path)
    return parameters, metrics, trajectories, residuals


def trait_bins(z, p, q, beta, kappa):
    """Exact moment-flux contributions; squared scores are averaged before binning."""
    p, q = probabilities(p), probabilities(q)
    if np.any((p == 0) & (q > 0)):
        raise ValueError("stored support failure prevents conditional reweighting analysis")
    state = moments(z, p)
    x = z - state["mu"]
    bins = np.searchsorted([-1., 0., 1., 2., 3.], x / np.sqrt(state["V"]), side="right")
    result = []
    for index in range(6):
        mask = bins == index
        mass = float(p[mask].sum())
        if mass == 0:
            continue
        zm = float(p[mask] @ z[mask] / mass)
        q_part = float((q[mask] - p[mask]) @ (x[mask] ** 2))
        linear = float(beta * (p[mask] @ (x[mask] ** 3)))
        result.append({"bin": index, "mass": mass, "score_mean": zm,
                       "omega_mean": float(q[mask].sum() / mass), "linear_prediction": 1 + beta * (zm - state["mu"]),
                       "Q_contribution": q_part, "beta_M3_contribution": linear,
                       "Q_linear_residual": q_part - linear, "Q_flux_residual": q_part - kappa * linear})
    return result


def early_diagnostics(archive, runs):
    records = []
    with np.load(archive, allow_pickle=False) as data:
        seeds, scores, weights = data["seeds"], data["particle_scores"], data["particle_weights"]
        for rid, rows in sorted(runs.items()):
            matches = np.flatnonzero(seeds == int(rid.rsplit("-", 1)[-1]))
            if len(matches) != 1:
                raise ValueError(f"archive seed does not match {rid}")
            ri, kappa = int(matches[0]), fit_kappa(rows)
            z = np.asarray(scores[ri], dtype=float)
            by_step = {r["step"]: r for r in rows}
            for step in EARLY_STEPS:
                if step not in by_step or step + 1 >= weights.shape[1]:
                    continue
                p, q = probabilities(weights[ri, step]), probabilities(weights[ri, step + 1])
                state, row = moments(z, p), by_step[step]
                x = z - state["mu"]
                computed = [state["mu"], state["V"], (q - p) @ x, (q - p) @ (x * x)]
                if not np.allclose(computed, [row[k] for k in ("mu", "V", "C", "Q")], rtol=1e-7, atol=1e-12):
                    raise ValueError(f"archive moments/fluxes do not match table for {rid} at {step}")
                records.extend({"run_id": rid, "step": step, **r} for r in trait_bins(z, p, q, row["beta"], kappa))
    return records


def report(output, binary, neural, parameters, metrics, trajectories, stability, bins):
    figures = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name, caption):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=125, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        figures.append(f"<figure><img src='data:image/png;base64,{encoded}' alt='{name}'><figcaption>{caption}</figcaption></figure>")

    def trajectory(ax, rid, state, models):
        for model in models:
            sub = [r for r in trajectories if r["run_id"] == rid and r["model"] == model]
            ax.plot([r["step"] for r in sub], [r[f"{state}_generated"] for r in sub], label=model)
        ax.plot([r["step"] for r in sub], [r[f"{state}_observed"] for r in sub], "k--", label="observed")
        ax.set(xlabel="step", ylabel=state)
        ax.legend(fontsize=7)

    def errors(ax, runs, field, models):
        for model in models:
            sub = [next(r for r in metrics if r["run_id"] == rid and r["model"] == model) for rid in sorted(runs)]
            ax.plot(range(len(runs)), [r[field] if r[field] is not None else np.nan for r in sub], "o-", label=model)
        ax.set(xticks=range(len(runs)), xticklabels=[rid.rsplit("-", 1)[-1] for rid in sorted(runs)], ylabel=field)
        ax.legend(fontsize=7)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    overlap = [max(values(rs, "T").min() for rs in binary.values()), min(values(rs, "T").max() for rs in binary.values())]
    for i, rid in enumerate(sorted(binary)):
        r = next(r for r in parameters if r["run_id"] == rid and r["model"] == "state_own")
        grid = np.linspace(*overlap, 100)
        axes[0, 0].plot(grid, polyval(grid, [r[f"c{j}"] for j in range(3)]), label=rid.rsplit("-", 1)[-1])
        axes[0, 1].scatter(i, r["upper_root"], color=f"C{i}")
    axes[0, 0].set(xlabel="T (shared observed range)", ylabel="S(T)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(xticks=range(len(binary)), xticklabels=[r.rsplit("-", 1)[-1] for r in sorted(binary)], ylabel="Own-run upper root")
    trajectory(axes[1, 0], sorted(binary)[0], "mu", ["state_own", "state_loo", "time_loo"])
    errors(axes[1, 1], binary, "mu_rmse", ["state_own", "state_loo", "time_loo"])
    save(fig, "binary", "Own-run curves and roots describe heterogeneity. LOO curves use other runs only. First recorded run is shown at left; every run is scored at right. Seed changes include environment changes.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    first = sorted(neural)[0]
    rs = [r for r in neural[first] if r.get("Q") is not None]
    x, q = values(rs, "beta") * values(rs, "M3"), values(rs, "Q")
    axes[0].scatter(x, q, s=7, alpha=.4)
    grid = np.array([x.min(), x.max()])
    axes[0].plot(grid, grid, "--", label="κ = 1")
    axes[0].plot(grid, fit_kappa(rs) * grid, label="own κ")
    axes[0].set(xlabel="Measured β M3", ylabel="Measured Q", title=first)
    axes[0].legend()
    labels = ["whole_run"] + [f"{a}:{b}" for a, b in WINDOWS]
    for rid in sorted(neural):
        by_window = {r["window"]: r for r in stability if r["run_id"] == rid}
        axes[1].plot(range(len(labels)), [by_window.get(label, {}).get("kappa", np.nan) for label in labels], "o-", label=rid.rsplit("-", 1)[-1])
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, ylabel="κ fitted within window")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=7)
    save(fig, "kappa", "κ minimizes sum(Q − κ β M3)² with no intercept, using measured moments. Window fits diagnose stability; they are not additional rollout models. Fitting weight shares appear below.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, step in enumerate((0, 25, 100)):
        bs = [r for r in bins if r["run_id"] == first and r["step"] == step]
        if not bs:
            continue
        z = [r["score_mean"] for r in bs]
        axes[0].plot(z, [r["omega_mean"] for r in bs], "o-", color=f"C{i}", label=f"step {step}")
        axes[0].plot(z, [r["linear_prediction"] for r in bs], "--", color=f"C{i}")
        axes[1].plot(z, [r["Q_linear_residual"] for r in bs], "o-", color=f"C{i}", label=f"step {step}")
    axes[0].set(xlabel="Mean trait value within bin", ylabel="Conditional mean w")
    axes[1].set(xlabel="Mean trait value within bin", ylabel="Bin contribution to Q − β M3")
    axes[1].axhline(0, color="grey", lw=.7)
    axes[0].legend()
    save(fig, "early_residuals", "First recorded neural run. Dashed curves are the uncorrected linear reference. Signed contributions use exact within-bin moments, not powers of bin means. CSVs contain all runs at steps 0, 10, 25, 50, 100, 200.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axs, state in zip(axes, ["mu", "V"]):
        trajectory(axs[0], first, state, ["uncorrected_own", "corrected_own", "loo_kappa", "loo_all"])
        errors(axs[1], neural, f"{state}_rmse", ["uncorrected_own", "corrected_own", "loo_kappa", "loo_all"])
    save(fig, "continuous", "Only loo_all transfers every fitted coefficient. loo_kappa transfers κ while β and γ still use the target run. All paths use generated states after initialization; failed paths receive no full-horizon RMSE.")

    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Closure transfer and flux correction</title>",
            "<style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{display:block;overflow:auto;border-collapse:collapse;font-size:.8rem}th,td{padding:.4rem;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>Development transfer and a one-parameter flux correction</h1>",
            "<p>Binary: compare quadratic S(T) and quadratic S(t), each fitted on the other whole runs by unweighted least squares. Own-run fits are descriptive references. LOO uses only the target's initial T during generation. This is development transfer on already inspected runs, not untouched prospective validation.</p>",
            "<p>Continuous: retain affine β(μ) and γ(μ). Fit κ from measured Q and βM3, without tuning trajectory error. Generate μ′ = μ + βV and V′ = V + κβγV<sup>3/2</sup> − (βV)². κ corrects Q only: it neither rescales the mean-selection law nor establishes linear conditional reweighting.</p>",
            "<p>Protocols: corrected_own fits every coefficient within the target run; shared_kappa pools κ across all runs but keeps own β,γ; loo_kappa estimates κ from the other runs but keeps own β,γ; loo_all fits β,γ,κ on the other runs. uncorrected_own uses own β,γ and κ = 1. Only loo_all is a complete coefficient-transfer test. Pooled fits weight observations equally; κ estimation consequently weights each run/window through its sum(βM3)².</p>",
            "<p>The original Q−βM3 remains separate from Q−κβM3. Corrected signed decomposition: Q−κβ̂M̂3 = (Q−κβM3) + κβ(M3−M̂3) + κ(β−β̂)M̂3. Relative norms use the same measured Q transitions; norms do not add. Signed sums on observed states are not a decomposition of autonomous endpoint error.</p>",
            "<p>Trait bins use standardized-score boundaries −1, 0, 1, 2, 3 with open tails. Bin contributions are Σ[(q−p)x²−βpx³], x=z−μ, so they sum to Q−βM3 exactly. Dashed conditional references remain 1+βx. Archives are checked against measured moments/fluxes before comparison.</p>"]
    html.extend(figures)
    for rid in sorted(set(binary) | set(neural)):
        html.append(f"<details><summary>{rid}</summary>")
        ps = [r for r in parameters if r["run_id"] == rid]
        html.append(table(ps, ["model", "c0", "c1", "c2", "upper_root"] if rid in binary else ["model", "beta0", "beta1", "gamma0", "gamma1", "kappa"]))
        html.append(table([r for r in metrics if r["run_id"] == rid], ["model", "status", "mu_rmse", "V_rmse"] +
                          (["Q_flux_residual_relative_l2", "Q_total_residual_relative_l2"] if rid in neural else [])))
        if rid in neural:
            html.append(table([r for r in stability if r["run_id"] == rid], ["window", "n", "kappa", "fit_weight_share", "correlation", "relative_Q_error_full_kappa"]))
        html.append("</details>")
    html.append("<p>Initial states are excluded from RMSE; missing terminal neural variance is omitted, while its mean is retained. Finite probabilities and nonnegative variance are required; no clipping is applied. Stop for discussion before more flexible closures or new experiments. Four figures are embedded; CSVs retain coefficients, metrics, trajectories, signed residuals, stability windows, and early trait bins at full precision.</p></html>")
    (output / "index.html").write_text("\n".join(html))


def run(table_path, archive, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    groups = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(table_path):
        if row["family"] in ("tabular_binary", "neural_continuous"):
            groups[row["family"]][row["run_id"]].append(row)
    binary, neural = groups["tabular_binary"], groups["neural_continuous"]
    for runs in (binary, neural):
        if len(runs) < 2:
            raise ValueError("need at least two runs from each controlled system for transfer")
        for rows in runs.values():
            rows.sort(key=lambda r: r["step"])
            validate_run(rows)
    bp, bm, bt = binary_transfer(binary)
    cp, cm, ct, residuals = continuous_transfer(neural)
    stability, bins = kappa_stability(neural), early_diagnostics(archive, neural)
    parameters, metrics, trajectories = bp + cp, bm + cm, bt + ct
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("parameters", parameters), ("metrics", metrics), ("trajectories", trajectories),
                       ("flux_residuals", residuals), ("kappa_windows", stability), ("early_trait_bins", bins)]:
        write_csv(output / f"{name}.csv", [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for k, v in r.items()} for r in rows])
    report(output, binary, neural, parameters, metrics, trajectories, stability, bins)
    print(f"Generated {len(metrics)} trajectories; report: {output / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--neural-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.table, args.neural_archive, args.output)
