"""Pooled predicted-vs-observed ΔT scatter across GRPO seeds -- the seed-robustness
hero figure. Each seed retrains from a different GRPO seed on the SAME frozen SAE
feature, so its (transition, feature) points are independent replicates of the SAME
claim: the sampled Price estimator cov(ω,s) recovers the directly-measured drift ΔT.

Pooling is legitimate because every point is one predicted-vs-observed pair and the
claim is that they lie on y=x regardless of seed; colouring by seed shows no single
run drives the fit. corr/slope are reported as mean±sd OVER seeds (each seed is one
replicate) rather than pooled, so the spread is the between-seed spread.

  uv run python -m analysis.aggregate_seeds \
      --config experiments/configs/config_real_lr1e-4.yaml \
      --runs real_lr1e-4 real_lr1e-4_s2 real_lr1e-4_s3
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    # Match Adil's figures: matplotlib default sans-serif (DejaVu Sans), base size 10,
    # all other text sizes at their defaults relative to font.size.
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Prediction = the omega_bar-corrected covariance form; fall back to naive price.
_pred = lambda r: r.get("cov", r["price"])
# distinct per-seed marker colours (avoid grey -- reserved for controls)
SEED_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _seed_arrays(jsonl_path):
    rows = [json.loads(l) for l in open(jsonl_path)]
    N_max = max(r["N"] for r in rows)
    rN = [r for r in rows if r["N"] == N_max]
    sig = [r for r in rN if not r["is_control"]]
    ctl = [r for r in rN if r["is_control"]]
    return {
        "N": N_max,
        "obs": np.array([r["direct_drift"] for r in sig]),
        "prd": np.array([_pred(r) for r in sig]),
        "obs_c": np.array([r["direct_drift"] for r in ctl]),
        "prd_c": np.array([_pred(r) for r in ctl]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="results-dir names, one per seed (first is the reference seed)")
    ap.add_argument("--out", default=None, help="output dir (default: <results>/plots_seeds)")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["OutputConfig"]["results_dir"])

    seeds = {}
    for name in args.runs:
        p = root / name / "price_eval.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        seeds[name] = _seed_arrays(p)

    # per-seed corr & slope (each seed = one replicate); report mean±sd over seeds.
    corrs, slopes, ctl_mags = [], [], []
    for s in seeds.values():
        corrs.append(np.corrcoef(s["obs"], s["prd"])[0, 1])
        slopes.append(np.polyfit(s["obs"], s["prd"], 1)[0])
        if len(s["prd_c"]):
            ctl_mags.append(np.abs(s["prd_c"]).mean())
    corrs, slopes = np.array(corrs), np.array(slopes)
    print(f"{'seed':>18}  {'corr':>6}  {'slope':>6}")
    for name, c, sl in zip(args.runs, corrs, slopes):
        print(f"{name:>18}  {c:6.3f}  {sl:6.3f}")
    print(f"{'mean±sd':>18}  {corrs.mean():.3f}±{corrs.std(ddof=1):.3f}"
          f"  {slopes.mean():.3f}±{slopes.std(ddof=1):.3f}")
    if ctl_mags:
        print(f"control |cov| (mean over seeds): {np.mean(ctl_mags):.4f}")

    # symmetric limits across all seeds' signal points
    allpts = np.concatenate([np.concatenate([s["obs"], s["prd"]]) for s in seeds.values()])
    lim = float(np.abs(np.concatenate([allpts, [0.0]])).max()) * 1.1

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.9, color="grey", zorder=0)
    ax.axhline(0, lw=0.5, color="grey", zorder=0); ax.axvline(0, lw=0.5, color="grey", zorder=0)
    # Encode both dimensions everywhere: colour = seed, marker fill = tracked vs control.
    seed_handles = []
    for i, (_, s) in enumerate(seeds.items()):
        col = SEED_COLORS[i % len(SEED_COLORS)]
        if len(s["obs_c"]):
            ax.scatter(s["obs_c"], s["prd_c"], s=20, facecolors="none", edgecolors=col,
                       linewidths=0.9, alpha=0.7, zorder=2)
        seed_handles.append(
            ax.scatter(s["obs"], s["prd"], s=22, color=col, alpha=0.75,
                       edgecolor="none", label=f"Seed {i + 1}", zorder=3)
        )
    # Marker-shape keys are colour-agnostic: open = control, filled = tracked.
    from matplotlib.lines import Line2D
    shape_leg = [
        Line2D([0], [0], marker="o", color="dimgrey", ls="none", ms=6,
               label="Tracked"),
        Line2D([0], [0], marker="o", mfc="none", mec="dimgrey", color="dimgrey",
               ls="none", ms=6, label="Control"),
    ]
    N = next(iter(seeds.values()))["N"]
    ax.set_title(f"Predicted vs. Observed ΔT ($N={N}$, {len(seeds)} Seeds)\n"
                 f"Correlation = {corrs.mean():.3f} ± {corrs.std(ddof=1):.3f}    "
                 f"Slope = {slopes.mean():.3f} ± {slopes.std(ddof=1):.3f}")
    ax.set_xlabel("Observed ΔT (Direct)")
    ax.set_ylabel("Predicted ΔT (Price, Cov)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    legend_handles = ([seed_handles[0], shape_leg[0], seed_handles[1],
                       shape_leg[1], seed_handles[2]]
                      if len(seed_handles) == 3 else seed_handles + shape_leg)
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.07))
    plt.tight_layout(rect=(0, 0.15, 1, 1))

    out_dir = Path(args.out) if args.out else root / "plots_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"seed_scatter.{ext}", dpi=150)
    plt.close(fig)
    print(f"done -> {out_dir}/seed_scatter.png")


if __name__ == "__main__":
    main()
