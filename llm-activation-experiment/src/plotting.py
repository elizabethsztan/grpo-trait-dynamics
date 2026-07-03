import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(jsonl_path):
    return [json.loads(l) for l in open(jsonl_path)]


def plot_from_jsonl(jsonl_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(jsonl_path)
    if not rows:
        return
    N_max = max(r["N"] for r in rows)
    feats = sorted({r["feature_id"] for r in rows})
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

    # grid_price: cumulative observed (direct) vs predicted (price) at N_max, per feature
    n = len(feats); cols = min(5, n); rows_g = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 2.6 * rows_g), squeeze=False)
    for ax, fid in zip(axes.flat, feats):
        fr = [r for r in rows if r["feature_id"] == fid and r["N"] == N_max]
        fr.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in fr]
        obs = np.cumsum([r["direct_drift"] for r in fr])
        pred = np.cumsum([r["price"] for r in fr])
        ctrl = fr[0]["is_control"] if fr else False
        ax.plot(steps, obs, color=colors[0], marker="o", ms=3, label="observed ΔT")
        ax.plot(steps, pred, color=colors[1], marker="s", ms=3, label="Price")
        ax.set_title(f"feat {fid}{' (ctrl)' if ctrl else ''}", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for ax in axes.flat[len(feats):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("GRPO step t"); fig.supylabel("cumulative trait change")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"grid_price.{ext}", dpi=150)
    plt.close(fig)

    # price_convergence: final-step price vs N, one line per feature, dashed = direct
    fig, ax = plt.subplots(figsize=(5, 4))
    last_step = max(r["step"] for r in rows)
    for i, fid in enumerate(feats):
        fr = [r for r in rows if r["feature_id"] == fid and r["step"] == last_step]
        fr.sort(key=lambda r: r["N"])
        Ns = [r["N"] for r in fr]
        ax.plot(Ns, [r["price"] for r in fr], marker="o", ms=3, color=colors[i % len(colors)])
        ax.axhline(fr[-1]["direct_drift"], ls="--", lw=0.8, color=colors[i % len(colors)])
    ax.set_xscale("log"); ax.set_xlabel("Price sample budget N"); ax.set_ylabel("final-step estimate")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"price_convergence.{ext}", dpi=150)
    plt.close(fig)

    # omega_diagnostics: mean_omega and ess vs step (at N_max)
    fr = [r for r in rows if r["N"] == N_max and r["feature_id"] == feats[0]]
    fr.sort(key=lambda r: r["step"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    a1.plot([r["step"] for r in fr], [r["mean_omega"] for r in fr], color=colors[0], marker="o", ms=3)
    a1.axhline(1.0, ls="--", lw=0.8, color="grey"); a1.set_ylabel(r"$\bar\omega$")
    a2.plot([r["step"] for r in fr], [r["ess"] for r in fr], color=colors[2], marker="o", ms=3)
    a2.set_ylabel("ESS"); a2.set_xlabel("GRPO step t")
    a2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"omega_diagnostics.{ext}", dpi=150)
    plt.close(fig)
