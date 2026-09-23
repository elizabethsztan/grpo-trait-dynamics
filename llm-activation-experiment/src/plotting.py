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

def _grid_layout(rows, features_path):
    """Panel labels/order for the per-feature grid, shared by grid_price and grid_price_seeds.

    Returns (labels, slots, cols, title_fn). One group per row: cols = the largest group,
    shorter groups padded with None so a row NEVER straddles two groups. That is what makes
    sharey='row' mean "one y-scale per group" -- per-panel autoscale magnifies each control's
    sampling noise to fill its frame and reads as false "signal", while a single global scale
    squashes the controls to invisibility against the larger negative-feature drifts.
    (Deriving cols from the group sizes rather than a fixed 5 keeps this correct when features
    are dropped -- e.g. 4/4/4 gives a clean 3x4 rather than a 5+5+2 that splits the groups.)
    """
    present = {r["feature_id"] for r in rows}
    # positive/negative lists in features.json are ranked by reward-correlation (rank 1 =
    # strongest), so "+#3 (rho=0.33)" = 3rd most reward-correlated; "-#1" = most anti-correlated.
    labels = {}
    if features_path.exists():
        fj = json.load(open(features_path))
        for i, e in enumerate(fj.get("positive", [])):
            labels[e["feature_id"]] = (f"+#{i + 1}", e.get("rho_val"))
        for i, e in enumerate(fj.get("negative", [])):
            labels[e["feature_id"]] = (f"−#{i + 1}", e.get("rho_val"))
        for e in fj.get("controls", []):
            labels[e["feature_id"]] = ("ctrl", e.get("rho_val"))
        groups = [[e["feature_id"] for e in fj.get(g, []) if e["feature_id"] in present]
                  for g in ("positive", "negative", "controls")]
        extra = sorted(present - {f for g in groups for f in g})
        if extra:
            groups.append(extra)
        groups = [g for g in groups if g]
    else:
        groups = None
    if groups:
        cols = max(len(g) for g in groups)
        slots = [f for g in groups for f in list(g) + [None] * (cols - len(g))]
    else:
        feats = sorted(present)
        cols = min(5, len(feats))
        slots = feats + [None] * (-len(feats) % cols)

    def title(fid, fr):
        lab, rho = labels.get(fid, (None, None))
        if lab is None:
            return f"feat {fid}{' (ctrl)' if (fr and fr[0]['is_control']) else ''}"
        return f"feat {fid}  {lab}" + (f" (ρ={rho:+.2f})" if rho is not None else "")

    return labels, slots, cols, title


def _cum_series(rows, fid, N, pred_of):
    """(steps, cumulative observed ΔT, cumulative Price prediction) for one feature at N."""
    fr = sorted((r for r in rows if r["feature_id"] == fid and r["N"] == N), key=lambda r: r["step"])
    steps = np.array([r["step"] for r in fr])
    obs = np.cumsum([r["direct_drift"] for r in fr])
    pred = np.cumsum([pred_of(r) for r in fr])
    return steps, obs, pred, fr


# Which sampled Price estimator to draw as "predicted ΔT":
#   "cov": raw omega_bar-corrected covariance  E[ωs] − E[ω]E[s]   (Adil-style)
#   "sn" : Hajek self-normalised  Σωs/Σω − E[s]  ==  cov / omega_bar
# The raw cov scales with the SAMPLE mean of ω, so it is only trustworthy when omega_bar≈1.
# Frozen-layer runs sit there (omega_bar≈1.05, ESS>400). All-layers LoRA moves the policy so
# far in its first steps that ω degenerates (ESS 6-12 of 512 at the 0->3 transition) and the
# sample omega_bar is essentially random (0.94 / 0.57 / 0.26 across three seeds), shrinking
# cov by that factor while the self-normalised form stays put. Hence estimator=None (auto)
# picks "sn" whenever the run logged the transmission term (i.e. it is an all-layers run)
# and "cov" otherwise -- so the frozen-layer headline figures are unchanged.
_ESTIMATORS = {
    "cov": (lambda r: r.get("cov", r["price"]), "cov"),
    "sn":  (lambda r: r.get("price_sn", r.get("cov", r["price"])), "self-norm"),
}


def _resolve_estimator(estimator, rows):
    if estimator is None:
        estimator = "sn" if any("transmission" in r for r in rows) else "cov"
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(_ESTIMATORS)}, got {estimator!r}")
    return (estimator, *_ESTIMATORS[estimator])


def plot_grid_seeds(jsonl_paths, out_dir, drop=None, stem="grid_price_seeds", estimator=None):
    """grid_price across GRPO seeds: each seed's cumulative curve drawn light, the
    across-seed mean in bold. Features/steps must match across seeds (same features.json,
    same Phase-3 schedule); features.json and panel order are taken from the FIRST run."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_paths = [Path(p) for p in jsonl_paths]
    drop = set(drop or ())
    per_seed = []
    for p in jsonl_paths:
        rows = [r for r in _load(p) if r["feature_id"] not in drop]
        per_seed.append(rows)
    if not per_seed or not per_seed[0]:
        return
    N_max = min(max(r["N"] for r in rows) for rows in per_seed)
    _, pred_of, est_label = _resolve_estimator(estimator, per_seed[0])
    labels, slots, cols, title = _grid_layout(per_seed[0], jsonl_paths[0].parent / "features.json")
    rows_g = len(slots) // cols
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 2.6 * rows_g),
                             squeeze=False, sharey="row")
    for ax, fid in zip(axes.flat, slots):
        if fid is None:
            ax.set_visible(False)
            continue
        series = [_cum_series(rows, fid, N_max, pred_of) for rows in per_seed]
        steps = series[0][0]
        if any(len(s[0]) != len(steps) or not np.array_equal(s[0], steps) for s in series):
            raise ValueError(f"feature {fid}: step schedules differ across seeds")
        obs = np.stack([s[1] for s in series]); pred = np.stack([s[2] for s in series])
        for o, pr in zip(obs, pred):
            ax.plot(steps, o, color=colors[0], lw=0.8, alpha=0.3)
            ax.plot(steps, pr, color=colors[1], lw=0.8, alpha=0.3)
        ax.plot(steps, obs.mean(0), color=colors[0], lw=2, marker="o", ms=3, label="observed ΔT")
        ax.plot(steps, pred.mean(0), color=colors[1], lw=2, marker="s", ms=3, label=f"Price ({est_label})")
        ax.set_title(title(fid, series[0][3]), fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes.flat[0].legend(frameon=False, fontsize=8,
                        title=f"bold = mean of {len(per_seed)} seeds", title_fontsize=8)
    fig.supxlabel("GRPO step t"); fig.supylabel("cumulative trait change")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"{stem}.{ext}", dpi=150)
    plt.close(fig)


def plot_from_jsonl(jsonl_path, out_dir, drop=None, estimator=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(jsonl_path)
    # drop: feature ids to exclude EVERYWHERE -- grid panels and the pooled scatter/decomp
    # fits alike -- so the reported corr/slope always describes exactly the features shown.
    drop = set(drop or ())
    if drop:
        rows = [r for r in rows if r["feature_id"] not in drop]
    if not rows:
        return
    N_max = max(r["N"] for r in rows)
    present = {r["feature_id"] for r in rows}
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    _, pred_of, est_label = _resolve_estimator(estimator, rows)

    labels, slots, cols, _title = _grid_layout(rows, Path(jsonl_path).parent / "features.json")
    feats = [f for f in slots if f is not None]
    rows_g = len(slots) // cols

    # grid_price: cumulative observed (direct) vs predicted (cov) at N_max, per feature
    fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 2.6 * rows_g),
                             squeeze=False, sharey="row")
    for ax, fid in zip(axes.flat, slots):
        if fid is None:
            ax.set_visible(False)
            continue
        fr = [r for r in rows if r["feature_id"] == fid and r["N"] == N_max]
        fr.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in fr]
        obs = np.cumsum([r["direct_drift"] for r in fr])
        pred = np.cumsum([pred_of(r) for r in fr])
        ax.plot(steps, obs, color=colors[0], marker="o", ms=3, label="observed ΔT")
        ax.plot(steps, pred, color=colors[1], marker="s", ms=3, label=f"Price ({est_label})")
        ax.set_title(_title(fid, fr), fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
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
    ax.set_xlabel("observed ΔT (direct)"); ax.set_ylabel(f"predicted ΔT (Price, {est_label})")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"price_scatter.{ext}", dpi=150)
    plt.close(fig)

    # price_decomp: only when run_price_eval was given --transmission (all-layers runs).
    # Tests the full Price identity  ΔT = cov (selection) + E[ω·Δs] (transmission).
    # For a frozen-trait run transmission≈0 and this collapses onto price_scatter, so
    # it's only drawn when the term is actually present.
    if any("transmission" in r for r in rN):
        sig = [r for r in rN if not r["is_control"]]
        obs = np.array([r["direct_drift"] for r in sig])
        sel = np.array([pred_of(r) for r in sig])                       # selection term
        trn = np.array([r["transmission"] for r in sig])
        full = sel + trn                                                # full prediction
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.5))

        # left: predicted vs observed -- selection-only vs selection+transmission.
        lim = float(np.abs(np.concatenate([obs, sel, full, [0.0]])).max()) * 1.1
        a1.plot([-lim, lim], [-lim, lim], ls="--", lw=0.8, color="grey", zorder=0)
        a1.axhline(0, lw=0.5, color="grey", zorder=0); a1.axvline(0, lw=0.5, color="grey", zorder=0)
        a1.scatter(obs, sel, s=22, facecolors="none", edgecolors=colors[1], linewidths=1.0,
                   label=f"selection only ({est_label})", zorder=2)
        a1.scatter(obs, full, s=22, color=colors[0], alpha=0.8,
                   label="selection + transmission", zorder=3)
        if obs.std() > 0:
            r_sel = np.corrcoef(obs, sel)[0, 1]; r_full = np.corrcoef(obs, full)[0, 1]
            a1.set_title(f"predicted vs observed ΔT (N={N_max})\n"
                         f"corr: {est_label}={r_sel:.2f} → {est_label}+trans={r_full:.2f}", fontsize=10.5)
        a1.set_xlabel("observed ΔT (direct)"); a1.set_ylabel("predicted ΔT")
        a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim); a1.set_aspect("equal")
        a1.legend(frameon=False, fontsize=8.5, loc="upper left")

        # right: how big is the transmission term relative to the selection term.
        m = float(np.abs(np.concatenate([sel, trn, [0.0]])).max()) * 1.1
        a2.axhline(0, lw=0.5, color="grey", zorder=0); a2.axvline(0, lw=0.5, color="grey", zorder=0)
        a2.scatter(sel, trn, s=22, color=colors[2], alpha=0.8, zorder=3)
        share = float(np.abs(trn).mean() / (np.abs(obs).mean() + 1e-12))
        a2.set_title(f"transmission vs selection\nmean |trans| / mean |ΔT| = {share:.2f}",
                     fontsize=10.5)
        a2.set_xlabel(f"selection  ({est_label})"); a2.set_ylabel("transmission  E[ω·Δs]")
        a2.set_xlim(-m, m); a2.set_ylim(-m, m); a2.set_aspect("equal")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(out_dir / f"price_decomp.{ext}", dpi=150)
        plt.close(fig)

    # grid_decomp: 4-line cumulative decomposition per feature, only when the transmission
    # term was logged (all-layers runs). Shows observed ΔT, the full prediction cov+trans
    # (should overlay observed), and the two components cov (selection) and trans separately
    # -- so features where transmission dominates (cov alone misses) are visible. Per-panel
    # autoscale (NOT sharey) so small-net-drift features aren't squashed by a big one sharing
    # the row; controls will therefore autoscale to their own noise (read the y-axis scale).
    # (Ported from the reconcile-llm branch, commit 9ae14e2, which never reached main.)
    if any("transmission" in r for r in rN):
        fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 2.6 * rows_g), squeeze=False)
        for ax, fid in zip(axes.flat, slots):
            if fid is None:
                ax.set_visible(False)
                continue
            steps, obs, cov, fr = _cum_series(rows, fid, N_max, pred_of)
            tr = np.cumsum([r.get("transmission", 0.0) for r in fr])
            ax.axhline(0, lw=0.5, color="grey", zorder=0)
            ax.plot(steps, obs, color=colors[0], lw=2.1, marker="o", ms=3, label="observed ΔT", zorder=4)
            ax.plot(steps, cov + tr, color=colors[2], ls="--", lw=1.6, marker="s", ms=2.5,
                    label=f"{est_label} + trans", zorder=3)
            ax.plot(steps, cov, color=colors[1], ls=":", lw=1.6, label=f"{est_label} (selection)", zorder=2)
            ax.plot(steps, tr, color=colors[3], ls="-.", lw=1.4, label="transmission", zorder=2)
            ax.set_title(_title(fid, fr), fontsize=9)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes.flat[0].legend(frameon=False, fontsize=7)
        fig.supxlabel("GRPO step t"); fig.supylabel("cumulative trait change")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(out_dir / f"grid_decomp.{ext}", dpi=150)
        plt.close(fig)

    # (N-convergence is shown properly by experiments/bands_from_pool.py -- bootstrap bands
    # over n=16..512 with the N=1024 reference line. A per-N line here is degenerate when the
    # config sweeps a single price budget, so it's intentionally not plotted.)

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
