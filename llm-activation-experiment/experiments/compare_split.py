"""Overlay the Price estimator computed on one split's rollouts against the held-out EVAL
trait curve -- the cross-split generalization test. Given a dumped rollout pool (from
run_price_eval --dump-pool --prompt-split <split>), computes cumulative cov(omega,s) per
transition and plots it against direct_eval (the reference trait curve) and cov_eval (the
in-distribution Price estimator), both read from the headline eval price_eval.jsonl.

If cov from a *different* split (e.g. train) threads direct_eval, cov from that split's
rollouts predicts the held-out trait drift.

  uv run python -m experiments.compare_split --config experiments/configs/config_real_lr1e-4.yaml \
      --pool-suffix _train --feature 25562
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
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _cov(w, s):
    return (w * s).mean(axis=-1) - w.mean(axis=-1) * s.mean(axis=-1)


def _labels(out):
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


def _eval_cumulatives(jsonl_path):
    # returns {feature_id: (steps(T+1,), direct_cum, cov_cum)} at N_max, cumulative from 0.
    rows = [json.loads(l) for l in open(jsonl_path)]
    N_max = max(r["N"] for r in rows)
    out = {}
    for fid in sorted({r["feature_id"] for r in rows}):
        fr = sorted([r for r in rows if r["feature_id"] == fid and r["N"] == N_max],
                    key=lambda r: r["step"])
        direct = np.concatenate([[0.0], np.cumsum([r["direct_drift"] for r in fr])])
        cov = np.concatenate([[0.0], np.cumsum([r.get("cov", r["price"]) for r in fr])])
        out[fid] = (np.arange(len(direct)), direct, cov)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-suffix", default="_train")   # split pool: pool<suffix>.npz
    ap.add_argument("--eval-jsonl", default="price_eval.jsonl")
    ap.add_argument("--feature", type=int, default=None)
    ap.add_argument("--split-label", default="train")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    labels = _labels(out)

    ev = _eval_cumulatives(out / args.eval_jsonl)
    d = np.load(out / f"pool{args.pool_suffix}.npz")
    omega, s, feat_ids = d["omega"], d["s"], d["feat_ids"]         # (T,N),(T,N,F),(F,)

    sel = list(range(len(feat_ids))) if args.feature is None else \
        [fi for fi in range(len(feat_ids)) if int(feat_ids[fi]) == args.feature]
    if not sel:
        raise SystemExit(f"--feature {args.feature} not in pool {list(map(int, feat_ids))}")
    band_dir = out / f"plots{args.pool_suffix}" / "cross_split"
    band_dir.mkdir(parents=True, exist_ok=True)

    # real GRPO-step x-axis for the pool line if recorded (step-stride runs), else index.
    xs_split = d["ckpt_steps"] if "ckpt_steps" in d else None
    for fi in sel:
        fid = int(feat_ids[fi])
        cov_split = np.concatenate([[0.0], np.cumsum(_cov(omega, s[:, :, fi]))])
        xs, direct_eval, cov_eval = ev.get(fid, (np.arange(len(cov_split)), None, None))
        x_split = xs_split if xs_split is not None else np.arange(len(cov_split))
        fig, ax = plt.subplots(figsize=(5.2, 4))
        ax.axhline(0, ls="--", lw=0.6, color="grey", zorder=0)
        if direct_eval is not None:
            ax.plot(xs, direct_eval, color="tab:blue", lw=2.2, label="direct ΔT (eval, target)")
            ax.plot(xs, cov_eval, color="tab:green", ls="--", lw=1.6, marker="s", ms=3,
                    label="cov (eval, in-dist)")
        ax.plot(x_split, cov_split, color="tab:orange", ls=":", lw=2.2,
                marker="o", ms=3, label=f"cov ({args.split_label}, cross-split)")
        ax.set_xlabel("GRPO step t"); ax.set_ylabel("cumulative trait change")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_title(f"feat {fid}  {labels.get(fid, '')}", fontsize=11)
        ax.legend(frameon=False)
        plt.tight_layout()
        stem = band_dir / f"feat_{fid}"
        for ext in ("png", "pdf"):
            plt.savefig(f"{stem}.{ext}", dpi=150)
        plt.close(fig)
        # numeric summary: corr of cov_split vs direct_eval across transitions
        if direct_eval is not None:
            c = np.corrcoef(cov_split, direct_eval)[0, 1]
            print(f"feat {fid:6d}  {labels.get(fid,''):16}  corr(cov_{args.split_label}, direct_eval)={c:.3f}")
    print(f"done -> {band_dir}")


if __name__ == "__main__":
    main()
