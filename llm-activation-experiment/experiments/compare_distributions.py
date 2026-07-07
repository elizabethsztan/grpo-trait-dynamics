"""Cross-distribution Price test. Overlays cumulative trait curves from a NEW measurement
distribution D' (e.g. SVAMP) against the OLD training distribution (GSM8K), using the same
GSM8K-trained checkpoints. Two questions:

  Q1 (identity holds on D'):   direct ΔT_{D'}  vs  cov_{D'}          -- should track (identity)
  Q2 (cross-dist transfer):    direct ΔT_{D'}  vs  cov_{GSM8K}       -- tracks IFF trait drift
                                                                        is distribution-invariant

Both read the per-transition jsonls written by run_price_eval (--out-suffix distinguishes
runs). Cumulative curves use each row's step_end for a correct x-axis (falls back to step+1
for pre-step_end jsonls, valid at stride 1).

  uv run python -m experiments.compare_distributions \
      --config experiments/configs/config_real_lr1e-4.yaml \
      --new-jsonl price_eval_svamp.jsonl --ref-jsonl price_eval.jsonl \
      --new-label svamp --ref-label gsm8k --feature 25562
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


def _cumulatives(jsonl_path):
    # {feature_id: (x(T+1,), direct_cum, cov_cum)}; x uses step_end (fallback step+1).
    rows = [json.loads(l) for l in open(jsonl_path)]
    N_max = max(r["N"] for r in rows)
    out = {}
    for fid in sorted({r["feature_id"] for r in rows}):
        fr = sorted([r for r in rows if r["feature_id"] == fid and r["N"] == N_max],
                    key=lambda r: r["step"])
        ends = [r.get("step_end", r["step"] + 1) for r in fr]
        x = np.array([fr[0]["step"]] + ends, dtype=float)
        direct = np.concatenate([[0.0], np.cumsum([r["direct_drift"] for r in fr])])
        cov = np.concatenate([[0.0], np.cumsum([r.get("cov", r["price"]) for r in fr])])
        out[fid] = (x, direct, cov)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--new-jsonl", required=True)
    ap.add_argument("--ref-jsonl", default="price_eval.jsonl")
    ap.add_argument("--new-label", default="svamp")
    ap.add_argument("--ref-label", default="gsm8k")
    ap.add_argument("--feature", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    labels = _labels(out)

    new = _cumulatives(out / args.new_jsonl)
    ref = _cumulatives(out / args.ref_jsonl)
    fids = [args.feature] if args.feature is not None else sorted(new)
    band_dir = out / "plots_crossdist"
    band_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'feat':>6}  {'label':16}  {'Q1 corr':>8}  {'Q2 corr':>8}")
    for fid in fids:
        if fid not in new:
            raise SystemExit(f"feature {fid} not in {args.new_jsonl}")
        xN, dN, cN = new[fid]               # D': x, direct, cov
        xR, dR, cR = ref[fid]               # GSM8K: x, direct, cov
        cR_at_N = np.interp(xN, xR, cR)     # GSM8K cov cumulative sampled at D' steps
        # Q1: cov_D' tracks direct_D'.  Q2: cov_GSM8K tracks direct_D'.
        q1 = np.corrcoef(cN, dN)[0, 1] if dN.std() > 0 else float("nan")
        q2 = np.corrcoef(cR_at_N, dN)[0, 1] if dN.std() > 0 else float("nan")
        print(f"{fid:6d}  {labels.get(fid,''):16}  {q1:8.3f}  {q2:8.3f}")

        fig, ax = plt.subplots(figsize=(5.4, 4.1))
        ax.axhline(0, ls="--", lw=0.6, color="grey", zorder=0)
        ax.plot(xN, dN, color="tab:blue", lw=2.4, marker="o", ms=4,
                label=f"direct ΔT ({args.new_label})")
        ax.plot(xN, cN, color="tab:green", ls="--", lw=1.7, marker="s", ms=3,
                label=f"cov ({args.new_label}, in-dist)  [Q1]")
        ax.plot(xR, cR, color="tab:orange", ls=":", lw=2.2,
                label=f"cov ({args.ref_label}, cross-dist)  [Q2]")
        ax.set_xlabel("GRPO step t"); ax.set_ylabel("cumulative trait change")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_title(f"feat {fid}  {labels.get(fid,'')}", fontsize=11)
        ax.legend(frameon=False)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(band_dir / f"feat_{fid}.{ext}", dpi=150)
        plt.close(fig)
    print(f"done -> {band_dir}")


if __name__ == "__main__":
    main()
