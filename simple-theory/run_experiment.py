import argparse
import json
import logging
import shutil
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

REWARD_BLOCK = {
    "trait_drives_reward": "TraitDrivesRewardConfig",
    "hidden_quality": "HiddenQualityConfig",
}


def cfg_value(cfg, key, default):
    value = cfg.get(key)
    return default if value is None else value


def select_reward_cfg(cfg, mode):
    return cfg[REWARD_BLOCK[mode]]


def build_policy(policy_cfg, reward_cfg, seed):
    mode = policy_cfg["mode"]
    kwargs = {
        "N": policy_cfg["N"],
        "K": policy_cfg["K"],
        "G": policy_cfg["G"],
        "mode": mode,
        "alpha": cfg_value(reward_cfg, "alpha", 1.0),
        "seed": seed,
    }
    if mode == "trait_drives_reward":
        kwargs["rho"] = reward_cfg["rho"]
        kwargs["b"] = reward_cfg["b"]
    elif mode == "hidden_quality":
        kwargs["gamma"] = cfg_value(reward_cfg, "gamma", 0.5)
        kwargs["p"] = cfg_value(reward_cfg, "p", 0.5)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return Policy(**kwargs)


def run_single(policy_cfg, reward_cfg, train_cfg, seed):
    policy = build_policy(policy_cfg, reward_cfg, seed)
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


def run_config(policy_cfg, reward_cfg, train_cfg, base_seed, num_runs, label=""):
    prefix = f"{label} " if label else ""
    trait_runs = []
    reward_runs = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        trait_curve, reward_curve = run_single(policy_cfg, reward_cfg, train_cfg, seed)
        trait_runs.append(trait_curve)
        reward_runs.append(reward_curve)
        LOGGER.info(f"{prefix}seed {seed} ({run_idx + 1}/{num_runs}): "
                    f"T_0={trait_curve[0]:.4f} -> T_final={trait_curve[-1]:.4f}, "
                    f"R_0={reward_curve[0]:.4f} -> R_final={reward_curve[-1]:.4f}")

    trait_runs = np.array(trait_runs)
    reward_runs = np.array(reward_runs)
    return {
        "steps_axis": np.arange(trait_runs.shape[1]),
        "trait_mean": trait_runs.mean(axis=0),
        "trait_sem": trait_runs.std(axis=0) / np.sqrt(num_runs),
        "reward_mean": reward_runs.mean(axis=0),
        "reward_sem": reward_runs.std(axis=0) / np.sqrt(num_runs),
    }


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
    mode = policy_cfg["mode"]
    reward_cfg = select_reward_cfg(cfg, mode)
    train_cfg = cfg["TrainConfig"]
    output_cfg = cfg["OutputConfig"]

    result = run_config(policy_cfg, reward_cfg, train_cfg, train_cfg["seed"], train_cfg["num_runs"])

    steps_axis = result["steps_axis"]
    trait_mean = result["trait_mean"]
    trait_sem = result["trait_sem"]
    reward_mean = result["reward_mean"]
    reward_sem = result["reward_sem"]

    run_dir = Path(output_cfg["results_dir"]) / output_cfg["name"]
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, run_dir / "config.yaml")

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "trait_mean": [round(float(v), 4) for v in trait_mean],
            "trait_sem": [round(float(v), 4) for v in trait_sem],
            "reward_mean": [round(float(v), 4) for v in reward_mean],
            "reward_sem": [round(float(v), 4) for v in reward_sem],
        }, f, indent=2)
    LOGGER.info(f"saved metrics to {metrics_path}")

    colors = [prop["color"] for prop in plt.rcParams["axes.prop_cycle"]]

    plot_trait(steps_axis, trait_mean, trait_sem, colors[0], plots_dir, "expected_trait")
    plot_trait_and_reward(
        steps_axis,
        [(r"Trait $T_t$", trait_mean, trait_sem), (r"Reward $R_t$", reward_mean, reward_sem)],
        colors, plots_dir, "trait_and_reward",
    )
    LOGGER.info(f"saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
