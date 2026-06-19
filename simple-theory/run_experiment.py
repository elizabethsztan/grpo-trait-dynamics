import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.tabular_policy import Policy

LOGGER = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def run_single(policy_cfg, train_cfg, seed):
    policy = Policy(
        N=policy_cfg["N"],
        K=policy_cfg["K"],
        G=policy_cfg["G"],
        rho=policy_cfg["rho"],
        b=policy_cfg["b"],
        alpha=policy_cfg["alpha"],
        seed=seed,
    )
    policy.init_env()

    steps = train_cfg["steps"]
    batch_size = train_cfg["batch_size"]
    eta = train_cfg["eta"]

    trait_curve = np.zeros(steps + 1)
    reward_curve = np.zeros(steps + 1)
    trait_curve[0] = policy.get_T()
    reward_curve[0] = policy.expected_reward()

    for step in range(steps):
        policy.grpo_step(batch_size=batch_size, eta=eta)
        trait_curve[step + 1] = policy.get_T()
        reward_curve[step + 1] = policy.expected_reward()

    return trait_curve, reward_curve


def plot_trait(steps_axis, trait_mean, trait_sem, color, output_dir, stem):
    fig, ax = plt.subplots(figsize=(5, 4))
    markevery = max(1, len(steps_axis) // 10)

    ax.fill_between(steps_axis, np.maximum(trait_mean - trait_sem, 0), trait_mean + trait_sem,
                    alpha=0.25, color=color, zorder=1)
    ax.plot(steps_axis, trait_mean, color=color, lw=1.5,
            marker="o", markersize=4, markevery=markevery, zorder=2)

    ax.axhline(trait_mean[0], color="grey", ls="--", lw=1, zorder=0)
    ax.set_xlabel(r"GRPO step $t$")
    ax.set_ylabel(r"Expected trait $T_t$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}.png", dpi=150)
    plt.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_trait_and_reward(steps_axis, series, colors, output_dir, stem):
    fig, ax = plt.subplots(figsize=(5, 4))
    markevery = max(1, len(steps_axis) // 10)

    for i, (label, mean, sem) in enumerate(series):
        ax.fill_between(steps_axis, np.maximum(mean - sem, 0), mean + sem,
                        alpha=0.25, color=colors[i], zorder=1)
    for i, (label, mean, sem) in enumerate(series):
        ax.plot(steps_axis, mean, label=label, color=colors[i], lw=1.5,
                marker="o", markersize=4, markevery=markevery, zorder=2)

    ax.set_xlabel(r"GRPO step $t$")
    ax.set_ylabel("Level")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}.png", dpi=150)
    plt.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy_cfg = cfg["PolicyConfig"]
    train_cfg = cfg["TrainConfig"]
    output_cfg = cfg["OutputConfig"]

    num_runs = train_cfg["num_runs"]
    base_seed = train_cfg["seed"]

    trait_runs = []
    reward_runs = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        LOGGER.info(f"running seed {seed} ({run_idx + 1}/{num_runs})")
        trait_curve, reward_curve = run_single(policy_cfg, train_cfg, seed)
        trait_runs.append(trait_curve)
        reward_runs.append(reward_curve)
        LOGGER.info(f"seed {seed}: T_0={trait_curve[0]:.4f} -> T_final={trait_curve[-1]:.4f}, "
                    f"R_0={reward_curve[0]:.4f} -> R_final={reward_curve[-1]:.4f}")

    trait_runs = np.array(trait_runs)
    reward_runs = np.array(reward_runs)
    steps_axis = np.arange(trait_runs.shape[1])

    trait_mean = trait_runs.mean(axis=0)
    trait_sem = trait_runs.std(axis=0) / np.sqrt(num_runs)
    reward_mean = reward_runs.mean(axis=0)
    reward_sem = reward_runs.std(axis=0) / np.sqrt(num_runs)

    output_dir = Path(output_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rho = policy_cfg["rho"]
    metrics_path = output_dir / f"metrics_rho-{rho}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": cfg,
            "trait_mean": [round(float(v), 4) for v in trait_mean],
            "trait_sem": [round(float(v), 4) for v in trait_sem],
            "reward_mean": [round(float(v), 4) for v in reward_mean],
            "reward_sem": [round(float(v), 4) for v in reward_sem],
        }, f, indent=2)
    LOGGER.info(f"saved metrics to {metrics_path}")

    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

    plot_trait(steps_axis, trait_mean, trait_sem, colors[0], output_dir, f"trait_rho-{rho}")
    plot_trait_and_reward(
        steps_axis,
        [(r"Trait $T_t$", trait_mean, trait_sem), (r"Reward $R_t$", reward_mean, reward_sem)],
        colors, output_dir, f"trait_reward_rho-{rho}",
    )
    LOGGER.info(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
