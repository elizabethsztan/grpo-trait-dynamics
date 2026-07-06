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
    present = {r["feature_id"] for r in rows}
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    # Prediction = the omega_bar-corrected covariance form (Adil-style); fall back to
    # the naive "price" for old runs that predate the cov field.
    pred_of = lambda r: r.get("cov", r["price"])

    # Feature labels + panel order from features.json (same dir), if available. Its
    # positive/negative lists are ranked by reward-correlation (rank 1 = strongest), so
    # "+#3 (rho=0.33)" = 3rd most reward-correlated feature; "-#1" = most anti-correlated.
    labels = {}
    fpath = Path(jsonl_path).parent / "features.json"
    if fpath.exists():
        fj = json.load(open(fpath))
        for i, e in enumerate(fj.get("positive", [])):
            labels[e["feature_id"]] = (f"+#{i + 1}", e.get("rho_val"))
        for i, e in enumerate(fj.get("negative", [])):
            labels[e["feature_id"]] = (f"−#{i + 1}", e.get("rho_val"))
        for e in fj.get("controls", []):
            labels[e["feature_id"]] = ("ctrl", e.get("rho_val"))
        order = ([e["feature_id"] for e in fj.get("positive", [])]
                 + [e["feature_id"] for e in fj.get("negative", [])]
                 + [e["feature_id"] for e in fj.get("controls", [])])
        feats = [f for f in order if f in present] + sorted(present - set(order))
    else:
        feats = sorted(present)

    def _title(fid, fr):
        lab, rho = labels.get(fid, (None, None))
        if lab is None:
            return f"feat {fid}{' (ctrl)' if (fr and fr[0]['is_control']) else ''}"
        return f"feat {fid}  {lab}" + (f" (ρ={rho:+.2f})" if rho is not None else "")

    # grid_price: cumulative observed (direct) vs predicted (cov) at N_max, per feature
    n = len(feats); cols = min(5, n); rows_g = int(np.ceil(n / cols))
    # sharey='row': one y-scale per row (rows are grouped positive / negative / control),
    # so features are comparable within a group and each group uses its natural range --
    # per-panel autoscale otherwise magnifies each control's sampling noise to fill its
    # frame and reads as a false "signal", while a single global scale squashes controls
    # to invisibility against the larger negative-feature drifts.
    fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 2.6 * rows_g),
                             squeeze=False, sharey="row")
    for ax, fid in zip(axes.flat, feats):
        fr = [r for r in rows if r["feature_id"] == fid and r["N"] == N_max]
        fr.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in fr]
        obs = np.cumsum([r["direct_drift"] for r in fr])
        pred = np.cumsum([pred_of(r) for r in fr])
        ax.plot(steps, obs, color=colors[0], marker="o", ms=3, label="observed ΔT")
        ax.plot(steps, pred, color=colors[1], marker="s", ms=3, label="Price (cov)")
        ax.set_title(_title(fid, fr), fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for ax in axes.flat[len(feats):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("GRPO step t"); fig.supylabel("cumulative trait change")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"grid_price.{ext}", dpi=150)
    plt.close(fig)

    # price_scatter: pooled predicted (cov) vs observed (direct) per (feature, transition)
    # at N_max -- the single clearest validation view. y=x is perfect agreement.
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    rN = [r for r in rows if r["N"] == N_max]
    obs_t = np.array([r["direct_drift"] for r in rN if not r["is_control"]])
    prd_t = np.array([pred_of(r) for r in rN if not r["is_control"]])
    obs_c = np.array([r["direct_drift"] for r in rN if r["is_control"]])
    prd_c = np.array([pred_of(r) for r in rN if r["is_control"]])
    lim = float(np.abs(np.concatenate([obs_t, prd_t, [0.0]])).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.8, color="grey", zorder=0)
    ax.axhline(0, lw=0.5, color="grey", zorder=0); ax.axvline(0, lw=0.5, color="grey", zorder=0)
    if len(obs_c):
        ax.scatter(obs_c, prd_c, s=18, color="lightgrey", edgecolor="grey", label="control", zorder=2)
    ax.scatter(obs_t, prd_t, s=20, color=colors[1], alpha=0.8, label="tracked", zorder=3)
    if obs_t.std() > 0:
        r_ = np.corrcoef(obs_t, prd_t)[0, 1]; sl = np.polyfit(obs_t, prd_t, 1)[0]
        ax.set_title(f"predicted vs observed ΔT (N={N_max})\ncorr={r_:.2f}  slope={sl:.2f}", fontsize=11)
    ax.set_xlabel("observed ΔT (direct)"); ax.set_ylabel("predicted ΔT (Price, cov)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"price_scatter.{ext}", dpi=150)
    plt.close(fig)

    # price_convergence: final-step price vs N, one line per feature, dashed = direct
    fig, ax = plt.subplots(figsize=(5, 4))
    last_step = max(r["step"] for r in rows)
    for i, fid in enumerate(feats):
        fr = [r for r in rows if r["feature_id"] == fid and r["step"] == last_step]
        fr.sort(key=lambda r: r["N"])
        Ns = [r["N"] for r in fr]
        ax.plot(Ns, [pred_of(r) for r in fr], marker="o", ms=3, color=colors[i % len(colors)])
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
