"""Per-run closure hypotheses for saved binary output and continuous SAE traits."""

import argparse
import base64
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyval, polyvander

from .fit_controlled import BINARY, fit_curve, rms, rollout, table, values
from .io import read_jsonl, write_csv
from .transfer_controlled import fit_kappa
import matplotlib.pyplot as plt


def fit_binary_flux(rows, predictor, degree):
    t, x, c = values(rows, "T"), values(rows, predictor), values(rows, "C")
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(c)
    t, x, c = t[valid], x[valid], c[valid]
    if ((t < 0) | (t > 1)).any():
        raise ValueError("binary prevalence must be in [0, 1]")
    v = t * (1 - t)
    design = v[:, None] * polyvander(x, degree)
    coef, _, rank, _ = np.linalg.lstsq(design, c, rcond=None)
    if rank != degree + 1:
        raise ValueError("binary flux law is unidentified on measured states")
    return coef, valid, int(np.count_nonzero(v))


def observations(rows, binary):
    """Keep old-panel moments and independent direct means as separate series."""
    observed = np.full((len(rows) + 1, 3), np.nan)  # mean, variance, separate direct mean

    def put(i, j, value):
        if value is None:
            return
        if np.isfinite(observed[i, j]) and not np.isclose(observed[i, j], value):
            raise ValueError("conflicting measurements at the same checkpoint")
        observed[i, j] = value

    for i, r in enumerate(rows):
        put(i, 0, r.get("T") if binary else r.get("mu"))
        if binary:
            put(i + 1, 0, r.get("mu_next"))
        else:
            put(i, 1, r.get("V"))
            put(i, 2, r.get("direct_mu_old"))
            put(i + 1, 2, r.get("direct_mu_new"))
    return observed


def fit_series(rows):
    step = values(rows, "step")
    if (not np.isfinite(step).all() or (step != np.floor(step)).any() or (np.diff(step) != 1).any()
            or (values(rows, "step_end") != step + 1).any() or (values(rows, "interval") != 1).any()):
        raise ValueError("LLM comparison requires contiguous one-update transitions")
    if any(r.get("inspection_status") != "development" for r in rows):
        raise ValueError("LLM fitting is restricted to development runs")
    binary = rows[0]["family"] == "llm_binary"
    observed = observations(rows, binary)
    initial = observed[0, :1 if binary else 2]
    if not np.isfinite(initial).all() or (not binary and (initial < 0).any()):
        raise ValueError("measured initial state is unavailable or outside the trait support")
    base = {k: rows[0][k] for k in ("family", "run_id", "trait_id", "setting_id")}
    base["is_control"] = rows[0].get("is_control")
    if not binary:
        fields = ("mu", "V", "beta", "skewness", "M3", "C", "Q")
        valid = np.isfinite(np.column_stack([values(rows, key) for key in fields])).all(axis=1)
        valid &= values(rows, "V") > 0
        fit_rows = [r for r, keep in zip(rows, valid) if keep]
        b = fit_curve(values(fit_rows, "mu"), values(fit_rows, "beta"), 1)
        g = fit_curve(values(fit_rows, "mu"), values(fit_rows, "skewness"), 1)
        kappa = fit_kappa(fit_rows)
    parameters, metrics, trajectories, residuals = [], [], [], []
    for name, predictor, degree in BINARY if binary else [("affine_uncorrected", "mu", 1), ("affine_corrected", "mu", 1)]:
        if binary:
            b, valid, informative = fit_binary_flux(rows, predictor, degree)
            g, k = None, 1.
            v = values(rows, "T") * (1 - values(rows, "T"))
            parts = {"C_residual": values(rows, "C")[valid] - v[valid] * polyval(values(rows, predictor)[valid], b)}
        else:
            k = kappa if name == "affine_corrected" else 1.
            informative = int(valid.sum())
            mu, v, beta, m3, q = (values(rows, key)[valid] for key in ("mu", "V", "beta", "M3", "Q"))
            mhat, bhat = polyval(mu, g) * v ** 1.5, polyval(mu, b)
            parts = {"C_residual": values(rows, "C")[valid] - bhat * v,
                     "Q_linear_residual": q - beta * m3, "Q_flux_residual": q - k * beta * m3,
                     "Q_moment_residual": k * beta * (m3 - mhat),
                     "Q_selection_residual": k * (beta - bhat) * mhat,
                     "Q_total_residual": q - k * bhat * mhat}
        parameters.append({**base, "model": name, "predictor": predictor, "c0": b[0],
                           "c1": b[1] if len(b) > 1 else None, "c2": b[2] if len(b) > 2 else None,
                           "gamma0": g[0] if g is not None else None, "gamma1": g[1] if g is not None else None,
                           "kappa": k if not binary else None, "fit_points": int(valid.sum()), "informative_points": informative})
        path, status = rollout(initial, step.astype(int), b, g, time=binary and predictor == "step", kappa=k,
                               mean_bounds=None if binary else (0., np.inf))
        metric = {**base, "model": name, "status": status,
                  "completed_updates": int(np.isfinite(path[1:, 0]).sum()),
                  "mu_rmse": rms(path[1:, 0] - observed[1:, 0]) if status == "complete" else None,
                  "V_rmse": rms(path[1:, 1] - observed[1:, 1]) if not binary and status == "complete" else None,
                  "direct_mu_rmse": rms(path[1:, 0] - observed[1:, 2]) if not binary and status == "complete" else None,
                  "mu_score_points": int(np.isfinite(observed[1:, 0]).sum()),
                  "direct_score_points": int(np.isfinite(observed[1:, 2]).sum()), "C_rmse": rms(parts["C_residual"])}
        c_scale = rms(values(rows, "C")[valid])
        metric["C_relative_l2"] = metric["C_rmse"] / c_scale if c_scale else None
        if not binary:
            for key, array in parts.items():
                if key.startswith("Q_"):
                    metric[key + "_relative_l2"] = rms(array) / rms(q) if rms(q) else None
            metric["beta_rmse"] = rms(values(rows, "beta")[valid] - polyval(mu, b))
            metric["gamma_rmse"] = rms(values(rows, "skewness")[valid] - polyval(mu, g))
        metrics.append(metric)
        for i, t in enumerate(np.r_[step, step[-1] + 1]):
            trajectories.append({**base, "model": name, "step": int(t), "mu_observed": observed[i, 0],
                                 "mu_generated": path[i, 0], "V_observed": observed[i, 1],
                                 "V_generated": path[i, 1] if not binary else None, "direct_mu": observed[i, 2]})
        for i, t in enumerate(step[valid]):
            residuals.append({**base, "model": name, "step": int(t), **{key: array[i] for key, array in parts.items()}})
    return parameters, metrics, trajectories, residuals


def report(output, groups, parameters, metrics, trajectories):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def save(fig, name, caption):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=125, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        figures.append(f"<figure><img src='data:image/png;base64,{encoded}' alt='{name}'><figcaption>{caption}</figcaption></figure>")

    def subset(records, key):
        return [r for r in records if (r["family"], r["run_id"], r["trait_id"]) == key]

    def paths(ax, key, state, models):
        for name in models:
            rs = [r for r in subset(trajectories, key) if r["model"] == name]
            ax.plot([r["step"] for r in rs], [r[f"{state}_generated"] for r in rs], label=name)
        ax.plot([r["step"] for r in rs], [r[f"{state}_observed"] for r in rs], "ko", ms=3, label="measured panel")
        if state == "mu" and key[0] == "llm_continuous":
            ax.plot([r["step"] for r in rs], [r["direct_mu"] for r in rs], "x", color="grey", label="separate direct mean")
        ax.set(xlabel="step", ylabel="T" if key[0] == "llm_binary" else state)
        ax.legend(fontsize=7)

    binary_keys = [key for key in sorted(groups) if key[0] == "llm_binary"]
    above = [key for key in binary_keys if "above_hook" in groups[key][0]["setting_id"]]
    other = [key for key in binary_keys if key not in above]
    for keys, title in [(above, "above_hook"), (other, "other_output_settings")]:
        if not keys:
            continue
        key, fig = keys[0], plt.figure(figsize=(13, 3.8))
        axs = fig.subplots(1, 3)
        rows = groups[key]
        t = values(rows, "T")
        axs[0].scatter(t, values(rows, "C"), color="grey", label="measured C")
        grid = np.linspace(np.nanmin(t), np.nanmax(t), 200)
        for p in subset(parameters, key):
            if p["predictor"] == "T":
                coef = [p[k] for k in ("c0", "c1", "c2") if p[k] is not None]
                axs[0].plot(grid, grid * (1 - grid) * polyval(grid, coef), label=p["model"])
        axs[0].set(xlabel="T", ylabel="C", title=key[1])
        axs[0].legend(fontsize=7)
        paths(axs[1], key, "mu", [name for name, _, _ in BINARY])
        for i, (name, _, _) in enumerate(BINARY):
            ms = [m for k in keys for m in subset(metrics, k) if m["model"] == name]
            axs[2].scatter([i] * len(ms), [m["mu_rmse"] if m["mu_rmse"] is not None else np.nan for m in ms], s=20)
        axs[2].set(xticks=range(len(BINARY)), xticklabels=[r[0] for r in BINARY], ylabel="T RMSE (complete paths only)")
        axs[2].tick_params(axis="x", rotation=50)
        save(fig, title, "First run by ID in this group is illustrated; dots summarize separate runs. Time/state rivals share parameter counts and flux-fitting objectives. Failed paths have missing full-horizon scores and remain in the tables below.")

    sae_keys = [key for key in sorted(groups) if key[0] == "llm_continuous"]
    candidates = [key for key in sae_keys if key[2] == "25273"] or sae_keys
    key = next((k for k in candidates if k[1] == "sae-rerun-s2"), candidates[0])
    rows = groups[key]
    corrected = next(p for p in subset(parameters, key) if p["model"] == "affine_corrected")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
    mu = values(rows, "mu")
    grid = np.linspace(mu.min(), mu.max(), 100)
    for ax, y, coef in [(axs[0], "beta", [corrected["c0"], corrected["c1"]]),
                         (axs[1], "skewness", [corrected["gamma0"], corrected["gamma1"]])]:
        ax.scatter(mu, values(rows, y), s=18)
        ax.plot(grid, polyval(grid, coef))
        ax.set(xlabel="mu", ylabel=y)
    x, q = values(rows, "beta") * values(rows, "M3"), values(rows, "Q")
    axs[2].scatter(x, q, s=18)
    grid = np.array([x.min(), x.max()])
    axs[2].plot(grid, grid, "--", label="κ = 1")
    axs[2].plot(grid, corrected["kappa"] * grid, label="fitted κ")
    axs[2].set(xlabel="Measured beta M3", ylabel="Measured Q")
    axs[2].legend()
    save(fig, "sae_components", f"Run {escape(key[1])}, feature {escape(key[2])}. Feature 25273 is the first selected feature in the original pool; rerun s2 also has separate direct means. This choice is fixed before fitting, not chosen by fit quality.")

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, state in zip(axs[:2], ["mu", "V"]):
        paths(ax, key, state, ["affine_uncorrected", "affine_corrected"])
    runs = sorted({k[1] for k in sae_keys})
    features = sorted({k[2] for k in sae_keys}, key=int)
    complete = np.full((len(features), len(runs)), np.nan)
    for r in metrics:
        if r["family"] == "llm_continuous" and r["model"] == "affine_corrected":
            complete[features.index(r["trait_id"]), runs.index(r["run_id"])] = int(r["status"] == "complete")
    axs[2].imshow(complete, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axs[2].set(xticks=range(len(runs)), xticklabels=runs, yticks=range(len(features)), yticklabels=features,
               title="Corrected path: green complete / red stopped")
    axs[2].tick_params(axis="x", rotation=60)
    save(fig, "sae_trajectories", "Initial mean/variance come from the old-policy panel; no later observations reset the model. Only the two reruns have separate direct mean levels. Final pool moments are missing, and original rounded drift is not accumulated into a mean trajectory. Features remain nested within training runs.")

    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>LLM closure hypotheses</title>",
            "<style>body{max-width:1150px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}img{max-width:100%}figure{margin:2rem 0}figcaption{font-size:.9rem}table{display:block;overflow:auto;border-collapse:collapse;font-size:.8rem}th,td{padding:.4rem;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600;margin:1rem 0}</style>",
            "<h1>LLM closure hypotheses: per-run development fits</h1>",
            "<p>Binary: fit C ≈ T(1−T)S(T) or T(1−T)S(t) directly by ordinary least squares. No division by T(1−T) is used. Constant, affine, and quadratic laws retain matched time rivals and a common constant. Every fit uses only checkpoints with measured old T and C; missing T is never interpolated. Boundary rows contribute to residuals but have zero coefficient information. The same flux objective is used in all output settings.</p>",
            "<p>SAE: fit affine β(μ) and affine standardized skewness γ(μ) by unweighted least squares, and a through-origin κ from measured Q versus βM3. All component fits, residuals, and fit counts use the same finite rows with positive variance; trajectory scoring retains all measured states. Compare κ = 1 with fitted κ. Generate μ′ = μ + βV and V′ = V + κβγV<sup>3/2</sup> − (βV)². These are hypotheses; neither κ nor trajectory agreement establishes linear conditional reweighting.</p>",
            "<p>Every trajectory uses only its measured initial state after fitting its own run. Fits use the whole development trajectory, so these are reconstructions, not held-out predictions. Nonfinite states, invalid probabilities, negative SAE means, or negative variance stop generation without clipping. Failed paths receive no full-horizon RMSE. Scores exclude supplied initial states.</p>",
            "<p>Binary mean scores use actual sparse direct evaluations. SAE mean/variance scores use old-policy sample panels, while direct_mu_rmse separately uses independent aggregate mean levels where saved. The final sample-panel mean and variance remain missing. Mean initialization uses the old-policy panel even when separate direct means exist. Historical rounded drifts never create substitute states.</p>",
            "<p>These LLM estimates lack calibrated sampling uncertainty. S, β, γ, Q, and κ can be noisy, especially near boundaries or small variance; flux fitting does not remove errors in measured T. Features from a run are dependent, and no parameters are pooled. Treat fit differences and residuals descriptively.</p>",
            "<p>Signed Q residuals separate as Q−κβ̂M̂3 = (Q−κβM3) + κβ(M3−M̂3) + κ(β−β̂)M̂3. The original Q−βM3 remains separately exported. Relative L2 norms use the measured Q norm; component norms do not add. Parameters, metrics, residuals, and trajectories are saved in four CSVs.</p>"]
    html.extend(figures)
    for family, title in [("llm_binary", "Output runs"), ("llm_continuous", "SAE features within runs")]:
        html.append(f"<p><strong>{title}</strong></p>")
        for key in sorted(groups):
            if key[0] != family:
                continue
            html.append(f"<details><summary>{escape(key[1])} · feature {escape(key[2])}</summary>")
            html.append(table(subset(parameters, key), ["model", "is_control", "c0", "c1", "c2", "gamma0", "gamma1", "kappa", "fit_points", "informative_points"]))
            html.append(table(subset(metrics, key), ["model", "status", "mu_rmse", "V_rmse", "direct_mu_rmse", "C_relative_l2"] +
                              ([f"Q_{part}_residual_relative_l2" for part in ("linear", "flux", "moment", "selection", "total")] if family == "llm_continuous" else [])))
            html.append("</details>")
    html.append(f"<p>Stop for discussion before selecting models, adding flexibility, or planning confirmation training. {len(figures)} figures are embedded and saved separately as PNG/PDF. A complete path only means that generation stayed within the checked bounds; it does not establish an adequate fit.</p></html>")
    (output / "index.html").write_text("\n".join(html))


def run(table_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    groups = defaultdict(list)
    for r in read_jsonl(table_path):
        if r["family"] in ("llm_binary", "llm_continuous"):
            groups[(r["family"], r["run_id"], r["trait_id"])].append(r)
    if {key[0] for key in groups} != {"llm_binary", "llm_continuous"}:
        raise ValueError("supply both LLM systems from the diagnostic table")
    combined = [[], [], [], []]
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["step"])
        for target, records in zip(combined, fit_series(rows)):
            target.extend(records)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in zip(("parameters", "metrics", "trajectories", "residuals"), combined):
        write_csv(output / f"{name}.csv", [{k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                                          for k, v in r.items()} for r in records])
    report(output, groups, *combined[:3])
    print(f"Fitted {len(groups)} run/trait series; {len(combined[1])} paths; report: {output / 'index.html'}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.table, args.output)
