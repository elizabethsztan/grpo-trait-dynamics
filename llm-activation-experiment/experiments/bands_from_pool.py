"""Offline bootstrap error bands for the sampled Price estimator, from a dumped pool.

Reads pool<suffix>.npz (produced by run_price_eval --dump-pool: raw per-rollout omega and
trait scores s for every transition), and for each feature makes a 1xK small-multiples
figure -- one panel per sample budget n. Each panel overlays:
  * reference line: the full-pool (N) cumulative Cov estimate -- the low-variance "gt".
  * n-budget band:  median + 16-84th pct of the cumulative estimate, from R bootstrap
    resamples of n rollouts (WITH replacement) from the N-pool at each transition.
Bands tighten as n grows -- that shrinkage IS the convergence. Pure CPU/numpy, no GPU.

  uv run python -m experiments.bands_from_pool --config experiments/configs/<cfg>.yaml \
      --pool-suffix _conv --n-list 16,32,64,128,256,512 --reps 500
"""
import argparse, json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _cov(w, s):
    # plug-in covariance = mean(w*s) - mean(w)*mean(s); the Price estimator itself.
    return (w * s).mean(axis=-1) - w.mean(axis=-1) * s.mean(axis=-1)


def _labels(out):
    # feature_id -> "+#i (rho=..)" style label, from features.json (same as plotting.py).
    fp = out / "features.json"
    if not fp.exists():
        return {}
    fj = json.load(open(fp)); lab = {}
    for i, e in enumerate(fj.get("positive", [])):
        lab[e["feature_id"]] = f"+#{i + 1} (ρ={e.get('rho_val'):+.2f})"
    for i, e in enumerate(fj.get("negative", [])):
        lab[e["feature_id"]] = f"−#{i + 1} (ρ={e.get('rho_val'):+.2f})"
    for e in fj.get("controls", []):
        lab[e["feature_id"]] = f"ctrl (ρ={e.get('rho_val'):+.2f})"
    return lab


def _cum_bootstrap(omega, s_f, n, reps, rng):
    # omega (T, N), s_f (T, N) for one feature. Returns (reps, T+1) cumulative trajectories,
    # each = running sum of per-transition Cov on an independent n-resample of that transition.
    T, N = omega.shape
    drifts = np.empty((reps, T))
    for t in range(T):
        idx = rng.integers(0, N, size=(reps, n))                # (reps, n) with replacement
        drifts[:, t] = _cov(omega[t][idx], s_f[t][idx])         # (reps,)
    cum = np.cumsum(drifts, axis=1)                             # (reps, T)
    return np.concatenate([np.zeros((reps, 1)), cum], axis=1)   # prepend T_0 = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-suffix", default="_conv")
    ap.add_argument("--n-list", default="16,32,64,128,256,512")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    # --feature: render only this feature_id (the convergence test needs just one rho!=0
    # trait). Omit to render every feature in the pool.
    ap.add_argument("--feature", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    n_list = [int(x) for x in args.n_list.split(",")]

    d = np.load(out / f"pool{args.pool_suffix}.npz")
    omega, s = d["omega"], d["s"]                # (T, N), (T, N, F)
    feat_ids, is_ctrl = d["feat_ids"], d["is_control"]
    T, N, F = s.shape
    if max(n_list) > N:
        raise SystemExit(f"n-list max {max(n_list)} exceeds pool N={N}")
    labels = _labels(out)
    # real GRPO-step x-axis if the pool recorded it (step-stride runs); else transition index.
    xs = d["ckpt_steps"] if "ckpt_steps" in d else np.arange(T + 1)
    band_dir = out / f"plots{args.pool_suffix}" / "bands"
    band_dir.mkdir(parents=True, exist_ok=True)

    sel = list(range(F))
    if args.feature is not None:
        sel = [fi for fi in range(F) if int(feat_ids[fi]) == args.feature]
        if not sel:
            raise SystemExit(f"--feature {args.feature} not in pool {list(map(int, feat_ids))}")
    for fi in sel:
        fid = int(feat_ids[fi]); s_f = s[:, :, fi]                          # (T, N)
        # reference: full-pool cumulative estimate (N samples), starts at 0.
        ref = np.concatenate([[0.0], np.cumsum(_cov(omega, s_f))])          # (T+1,)
        rng = np.random.default_rng(args.seed + fid)
        fig, axes = plt.subplots(1, len(n_list), figsize=(2.7 * len(n_list), 2.9),
                                 squeeze=False, sharey=True)
        for ax, n in zip(axes[0], n_list):
            cum = _cum_bootstrap(omega, s_f, n, args.reps, rng)             # (reps, T+1)
            lo, med, hi = np.percentile(cum, [16, 50, 84], axis=0)
            ax.axhline(0, ls="--", lw=0.6, color="grey", zorder=0)
            ax.fill_between(xs, lo, hi, color="tab:orange", alpha=0.25, lw=0)
            ax.plot(xs, med, color="tab:orange", ls="--", lw=1.4,
                    label=r"Sampled $\sum\widehat{\mathrm{Cov}}(\omega,s)$")
            ax.plot(xs, ref, color="tab:blue", lw=1.8, label=f"Reference (N={N})")
            ax.set_title(f"$n={n}$", fontsize=13)
            ax.set_xlabel("step $t$")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        axes[0][0].set_ylabel("cumulative trait change")
        axes[0][0].legend(frameon=False, fontsize=9, loc="best")
        lab = labels.get(fid, "ctrl" if is_ctrl[fi] else "")
        fig.suptitle(f"feat {fid}  {lab}", fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        stem = band_dir / f"feat_{fid}"
        for ext in ("png", "pdf"):
            plt.savefig(f"{stem}.{ext}", dpi=150)
        plt.close(fig)
        print(f"wrote {stem}.png  ({lab})")
    print(f"done -> {band_dir}")


if __name__ == "__main__":
    main()
