import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from run_experiment import run_config, cfg_value

LOGGER = logging.getLogger(__name__)


def plot_grid(gammas, ps, results, colors, output_dir, stem):
    n_rows = len(gammas)
    n_cols = len(ps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 1.9 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for i, gamma in enumerate(gammas):
        for j, p in enumerate(ps):
            ax = axes[i][j]
            res = results[(i, j)]
            steps_axis = res["steps_axis"]
            trait_mean = res["trait_mean"]
            trait_sem = res["trait_sem"]
            reward_mean = res["reward_mean"]
            reward_sem = res["reward_sem"]

            ax.fill_between(steps_axis, trait_mean - trait_sem, trait_mean + trait_sem,
                            alpha=0.25, color=colors[0], zorder=1)
            ax.fill_between(steps_axis, reward_mean - reward_sem, reward_mean + reward_sem,
                            alpha=0.25, color=colors[1], zorder=1)
            ax.plot(steps_axis, trait_mean, color=colors[0], lw=1.2, zorder=2,
                    label=r"Trait $T_t$" if i == 0 and j == 0 else None)
            ax.plot(steps_axis, reward_mean, color=colors[1], lw=1.2, zorder=2,
                    label=r"Reward $R_t$" if i == 0 and j == 0 else None)
            ax.axhline(trait_mean[0], color="grey", ls="--", lw=0.8, zorder=0)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))

            if i == 0:
                ax.set_title(rf"$p$ = {p}")
            if j == 0:
                ax.set_ylabel(rf"$\gamma$ = {gamma}")
            if i == n_rows - 1:
                ax.set_xlabel(r"step $t$")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / f"{stem}.png", dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_price_grid_curves(gammas, ps, results, colors, output_dir, stem, pred_key, pred_label):
    # curves-only grid (no residual panel) of observed vs a chosen prediction (exact or sampled)
    n_rows = len(gammas)
    n_cols = len(ps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 1.9 * n_rows),
                             sharex=True, squeeze=False)

    for i, gamma in enumerate(gammas):
        for j, p in enumerate(ps):
            ax = axes[i][j]
            res = results[(i, j)]
            steps_axis = res["steps_axis"]
            obs = res["observed_cum_mean"]
            obs_sem = res["observed_cum_sem"]
            pred = res[pred_key + "_mean"]
            pred_sem = res[pred_key + "_sem"]

            ax.fill_between(steps_axis, obs - obs_sem, obs + obs_sem, alpha=0.25, color=colors[0], zorder=1)
            ax.fill_between(steps_axis, pred - pred_sem, pred + pred_sem, alpha=0.25, color=colors[1], zorder=1)
            ax.plot(steps_axis, obs, color=colors[0], lw=1.2, zorder=2,
                    label=r"Observed: $T_t - T_0$" if i == 0 and j == 0 else None)
            ax.plot(steps_axis, pred, color=colors[1], lw=1.2, ls="--", zorder=2,
                    label=pred_label if i == 0 and j == 0 else None)
            ax.axhline(0, color="grey", ls="--", lw=0.8, zorder=0)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))

            if i == 0:
                ax.set_title(rf"$p$ = {p}")
            if j == 0:
                ax.set_ylabel(rf"$\gamma$ = {gamma}")
            if i == n_rows - 1:
                ax.set_xlabel(r"step $t$")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / f"{stem}.png", dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_price_samples_grid(samples_list, results, colors, output_dir, stem):
    # 1-row grid, one cell per price-sample count: observed (= exact dT) vs the sampled
    # cumulative estimator. As n grows the sampled curve tightens onto the observed line.
    n_cols = len(samples_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols, 2.6),
                             sharex=True, sharey=True, squeeze=False)

    for k, ns in enumerate(samples_list):
        ax = axes[0][k]
        res = results[k]
        steps_axis = res["steps_axis"]
        obs = res["observed_cum_mean"]
        obs_sem = res["observed_cum_sem"]
        samp = res["sampled_cum_mean"]
        samp_sem = res["sampled_cum_sem"]

        ax.fill_between(steps_axis, obs - obs_sem, obs + obs_sem, alpha=0.25, color=colors[0], zorder=1)
        ax.fill_between(steps_axis, samp - samp_sem, samp + samp_sem, alpha=0.25, color=colors[1], zorder=1)
        ax.plot(steps_axis, obs, color=colors[0], lw=1.2, zorder=2,
                label=r"Observed: $T_t - T_0$" if k == 0 else None)
        ax.plot(steps_axis, samp, color=colors[1], lw=1.2, ls="--", zorder=2,
                label=r"Sampled $\sum \widehat{\mathrm{Cov}}(\omega, s)$" if k == 0 else None)
        ax.axhline(0, color="grey", ls="--", lw=0.8, zorder=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
        ax.set_title(rf"$n$ = {ns}")
        ax.set_xlabel(r"step $t$")

    axes[0][0].set_ylabel("Cumulative trait change")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_dir / f"{stem}.png", dpi=150)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def run_gamma_p_sweep(cfg, policy_cfg, neural_cfg, train_cfg, mode, base_seed, num_runs,
                      price_check, run_dir, colors):
    hq_sweep = cfg["HiddenQualityConfigSweep"]
    gammas = hq_sweep["gamma"]
    ps = hq_sweep["p"]
    alpha = cfg_value(cfg["HiddenQualityConfig"], "alpha", 1.0)
    price_samples = cfg.get("PriceCheckConfig", {}).get("samples", 512)

    results = {}
    final = {}
    for i, gamma in enumerate(gammas):
        for j, p in enumerate(ps):
            reward_cfg = {"gamma": gamma, "p": p, "alpha": alpha}
            res = run_config(policy_cfg, reward_cfg, train_cfg, base_seed, num_runs,
                             label=f"gamma={gamma},p={p}", price_check=price_check,
                             mode=mode, neural_cfg=neural_cfg, price_samples=price_samples)
            results[(i, j)] = res
            cell = {
                "trait_mean": round(float(res["trait_mean"][-1]), 4),
                "trait_sem": round(float(res["trait_sem"][-1]), 4),
                "reward_mean": round(float(res["reward_mean"][-1]), 4),
                "reward_sem": round(float(res["reward_sem"][-1]), 4),
            }
            if price_check:
                observed_total = res["observed_cum_mean"][-1]
                predicted_total = res["predicted_cum_mean"][-1]
                cell["selection_explained_frac"] = (
                    round(float(predicted_total / observed_total), 4) if observed_total != 0 else None)
            final[f"gamma={gamma},p={p}"] = cell

    plots_dir = run_dir / "plots_sweep"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"gammas": gammas, "ps": ps, "final": final}, f, indent=2)
    LOGGER.info(f"saved metrics to {metrics_path}")

    plot_grid(gammas, ps, results, colors, plots_dir, "grid_trait_reward")
    if price_check:
        plot_price_grid_curves(gammas, ps, results, colors, plots_dir, "grid_price_exact",
                               "predicted_cum", r"Exact $\sum \mathrm{Cov}(\omega, s)$")
        if "sampled_cum_mean" in next(iter(results.values())):
            samples = cfg.get("PriceCheckConfig", {}).get("samples", 512)
            plot_price_grid_curves(gammas, ps, results, colors, plots_dir, "grid_price_estimated",
                                   "sampled_cum", rf"Sampled $n={samples}$")
    LOGGER.info(f"saved grid plot to {plots_dir}")


def run_price_samples_sweep(cfg, policy_cfg, neural_cfg, train_cfg, mode, base_seed, num_runs,
                            run_dir, colors):
    if mode != "neural":
        raise ValueError("--sweep price-samples requires --mode neural "
                         "(tabular has no sampled estimator)")

    hq = cfg["HiddenQualityConfig"]  # scalar gamma/p — fixed across the sweep
    reward_cfg = {"gamma": hq["gamma"], "p": hq["p"], "alpha": cfg_value(hq, "alpha", 1.0)}
    samples_list = cfg["PriceCheckConfigSweep"]["samples"]

    # Training is bit-identical across n (price_samples only feeds the diagnostic sampled Cov,
    # never the weight update), so we re-run per n for simplicity; only the sampled curve varies.
    results = {}
    final = {}
    for k, ns in enumerate(samples_list):
        res = run_config(policy_cfg, reward_cfg, train_cfg, base_seed, num_runs,
                         label=f"samples={ns}", price_check=True,
                         mode=mode, neural_cfg=neural_cfg, price_samples=ns)
        results[k] = res
        observed = res["observed_cum_mean"][-1]
        sampled = res["sampled_cum_mean"][-1]
        final[f"samples={ns}"] = {
            "observed": round(float(observed), 4),
            "sampled": round(float(sampled), 4),
            "residual": round(float(sampled - observed), 4),
        }

    plots_dir = run_dir / "plots_sweep_price_samples"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics_price_samples.json"
    with open(metrics_path, "w") as f:
        json.dump({"samples": samples_list, "final": final}, f, indent=2)
    LOGGER.info(f"saved metrics to {metrics_path}")

    plot_price_samples_grid(samples_list, results, colors, plots_dir, "grid_price_samples")
    LOGGER.info(f"saved grid plot to {plots_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["tabular", "neural"], default="tabular")
    parser.add_argument("--sweep", choices=["gamma-p", "price-samples"], default="gamma-p",
                        help="axis to sweep: gamma-p grid, or the price-sample budget (neural only)")
    parser.add_argument("--no-price-check", action="store_false", dest="price_check",
                        help="disable the Price-equation check (on by default; ignored by price-samples)")
    parser.set_defaults(price_check=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy_cfg = cfg["TabularPolicyConfig"]
    if policy_cfg["mode"] != "hidden_quality":
        raise ValueError("run_sweep only supports the hidden_quality reward model")

    if args.mode == "neural":
        neural_cfg = cfg["NeuralPolicyConfig"]
        train_cfg = cfg["NeuralTrainConfig"]
    else:
        neural_cfg = None
        train_cfg = cfg["TrainConfig"]

    output_cfg = cfg["OutputConfig"]
    base_seed = train_cfg["seed"]
    num_runs = train_cfg["num_runs"]
    colors = [prop["color"] for prop in plt.rcParams["axes.prop_cycle"]]

    run_dir = Path(output_cfg["results_dir"]) / output_cfg["name"] / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    if args.sweep == "price-samples":
        run_price_samples_sweep(cfg, policy_cfg, neural_cfg, train_cfg, args.mode,
                                base_seed, num_runs, run_dir, colors)
    else:
        run_gamma_p_sweep(cfg, policy_cfg, neural_cfg, train_cfg, args.mode,
                          base_seed, num_runs, args.price_check, run_dir, colors)


if __name__ == "__main__":
    main()
