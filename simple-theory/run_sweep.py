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

            ax.fill_between(steps_axis, np.maximum(trait_mean - trait_sem, 0), trait_mean + trait_sem,
                            alpha=0.25, color=colors[0], zorder=1)
            ax.fill_between(steps_axis, np.maximum(reward_mean - reward_sem, 0), reward_mean + reward_sem,
                            alpha=0.25, color=colors[1], zorder=1)
            ax.plot(steps_axis, trait_mean, color=colors[0], lw=1.2, zorder=2,
                    label=r"Trait $T_t$" if i == 0 and j == 0 else None)
            ax.plot(steps_axis, reward_mean, color=colors[1], lw=1.2, zorder=2,
                    label=r"Reward $R_t$" if i == 0 and j == 0 else None)
            ax.axhline(trait_mean[0], color="grey", ls="--", lw=0.8, zorder=0)
            ax.set_ylim(0, 1)
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


def plot_price_grid(gammas, ps, results, colors, output_dir, stem):
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
            pred = res["predicted_cum_mean"]
            pred_sem = res["predicted_cum_sem"]

            ax.fill_between(steps_axis, obs - obs_sem, obs + obs_sem, alpha=0.25, color=colors[0], zorder=1)
            ax.fill_between(steps_axis, pred - pred_sem, pred + pred_sem, alpha=0.25, color=colors[1], zorder=1)
            obs_label = r"Observed: $T_t - T_0$"
            pred_label = (r"Predicted: $\sum_{\tau<t}\widehat{\Delta T}_\tau"
                          r" = \sum_{\tau<t}\frac{B}{N}\widehat{\mathrm{Cov}}_\tau(\omega, s)$")
            ax.plot(steps_axis, obs, color=colors[0], lw=1.2, zorder=2,
                    label=obs_label if i == 0 and j == 0 else None)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy_cfg = cfg["PolicyConfig"]
    mode = policy_cfg["mode"]
    if mode != "hidden_quality":
        raise ValueError("run_sweep only supports mode: hidden_quality")

    hq_cfg = cfg["HiddenQualityConfig"]
    gammas = hq_cfg["gamma"]
    ps = hq_cfg["p"]
    alpha = cfg_value(hq_cfg, "alpha", 1.0)

    train_cfg = cfg["TrainConfig"]
    output_cfg = cfg["OutputConfig"]
    base_seed = train_cfg["seed"]
    num_runs = train_cfg["num_runs"]
    price_check = cfg.get("PriceCheckConfig", {}).get("enabled", False)

    results = {}
    final = {}
    for i, gamma in enumerate(gammas):
        for j, p in enumerate(ps):
            reward_cfg = {"gamma": gamma, "p": p, "alpha": alpha}
            res = run_config(policy_cfg, reward_cfg, train_cfg, base_seed, num_runs,
                             label=f"gamma={gamma},p={p}", price_check=price_check)
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

    run_dir = Path(output_cfg["results_dir"]) / output_cfg["name"]
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, run_dir / "config.yaml")

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"gammas": gammas, "ps": ps, "final": final}, f, indent=2)
    LOGGER.info(f"saved metrics to {metrics_path}")

    colors = [prop["color"] for prop in plt.rcParams["axes.prop_cycle"]]
    plot_grid(gammas, ps, results, colors, plots_dir, "grid_trait_reward")
    if price_check:
        plot_price_grid(gammas, ps, results, colors, plots_dir, "grid_price_check")
    LOGGER.info(f"saved grid plot to {plots_dir}")


if __name__ == "__main__":
    main()
