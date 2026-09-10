"""Per-run plots and descriptive overlays, without fitting or model selection."""

from collections import defaultdict
from html import escape
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_jsonl, sha256, write_csv, write_json


def _key(row):
    return row["run_id"], row["trait_id"]


def _slug(value):
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:110]
    return readable + "_" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _pairs(rows, x, y):
    return [(r[x], r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]


def _scatter(ax, rows, x, y):
    pairs = _pairs(rows, x, y)
    if pairs:
        a = np.asarray(pairs)
        ax.scatter(a[:, 0], a[:, 1], s=10, alpha=.65)
    else:
        ax.text(.5, .5, "Unavailable in saved data", ha="center", transform=ax.transAxes, fontsize=9)
    ax.set(xlabel=x, ylabel=y)
    ax.grid(alpha=.2)


def _save(fig, folder, stem):
    paths = [folder / f"{stem}.{ext}" for ext in ("png", "pdf")]
    if any(p.exists() for p in paths):
        plt.close(fig)
        raise FileExistsError(f"refusing to overwrite {stem}")
    for path in paths:
        fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return f"plots/{stem}.png"


def report(output):
    output = Path(output)
    if (output / "report.md").exists() or (output / "plots").exists():
        raise FileExistsError("report already exists; use a new output directory")
    folder = output / "plots"
    folder.mkdir()
    rows = read_jsonl(output / "transitions.jsonl")
    conditional = read_jsonl(output / "conditional_bins.jsonl")
    checks = read_jsonl(output / "accounting.jsonl")
    grouped, cgroups, agroups = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        grouped[_key(r)].append(r)
    for r in conditional:
        cgroups[_key(r)].append(r)
    for r in checks:
        agroups[_key(r)].append(r)
    summary, figures = [], []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for index, (key, series) in enumerate(sorted(grouped.items())):
        series.sort(key=lambda r: r["step"])
        first = series[0]
        binary = first["family"] in ("tabular_binary", "llm_binary")
        coeff, state = ("S", "T") if binary else ("beta", "mu")
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
        axes = axs.ravel()
        for ax, (x, y) in zip(axes[:4], [(state, coeff), ("step", coeff), ("step", state), ("step", "V")]):
            _scatter(ax, series, x, y)
        _scatter(axes[4], series, "step", "Q_minus_beta_M3")
        if binary:
            axes[4].clear()
            axes[4].text(.5, .5, "Binary moment relation is algebraic;\nnot a test of conditional linearity.",
                         ha="center", va="center", transform=axes[4].transAxes)
            axes[4].set_axis_off()
        account = sorted(agroups[key], key=lambda r: r["step"])
        _scatter(axes[5], account, "step", "residual")
        axes[5].set_title("Accounting: observed change − C sum")
        gaps = sorted({r["interval"] for r in series})
        # Variable gaps are displayed explicitly, not joined into a per-update law.
        fig.suptitle(f"{key[0]} · trait {key[1]}\n{first['setting_id']} · transition gaps {gaps}")
        fig.text(.01, .005, first["uncertainty_status"].replace("_", " ") + "; no closure fitted", fontsize=8)
        fig.tight_layout(rect=(0, .025, 1, .91))
        image = _save(fig, folder, _slug("run-" + "-".join(key)))
        images = [image]
        if cgroups[key]:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            by_step = defaultdict(list)
            for r in cgroups[key]:
                by_step[r["step"]].append(r)
            for j, (step, bins) in enumerate(sorted(by_step.items())):
                x = [r["score_mean"] for r in bins]
                y = [r["omega_mean"] for r in bins]
                color = f"C{j}"
                axes[0].plot(x, y, "o-", color=color, label=f"step {step}", lw=1, ms=3)
                if all(r["linear_prediction"] is not None for r in bins):
                    axes[0].plot(x, [r["linear_prediction"] for r in bins], "--", color=color)
                    axes[1].plot(x, [r["conditional_residual"] for r in bins], "o-", color=color, ms=3)
            axes[0].set(xlabel="trait score (bin mean)", ylabel="mean raw likelihood ratio",
                        title="Solid: measured; dashed: 1 + beta (z − mu)")
            axes[1].set(xlabel="trait score (bin mean)", ylabel="conditional residual")
            axes[1].axhline(0, color="grey", lw=.7)
            axes[0].legend()
            fig.suptitle(f"{key[0]} · trait {key[1]} · early/middle/late transitions")
            fig.tight_layout()
            images.append(_save(fig, folder, _slug("conditional-" + "-".join(key))))
        qres = [abs(r["Q_minus_beta_M3"]) for r in series if r.get("Q_minus_beta_M3") is not None]
        summary.append({"run_id": key[0], "family": first["family"], "setting_id": first["setting_id"],
                        "trait_id": key[1], "transition_count": len(series),
                        "coefficient_count": sum(r.get(coeff) is not None for r in series),
                        "intervals": gaps, "max_abs_Q_minus_beta_M3": max(qres) if qres else None,
                        "uncertainty_status": first["uncertainty_status"], "figure": image})
        figures.append((f"{key[0]} · trait {key[1]}", images))
        if (index + 1) % 10 == 0:
            print(f"Rendered {index + 1}/{len(grouped)} individual trait series", flush=True)
    # Overlays preserve every run and separate settings, feature IDs, and interval lengths.
    overlays = defaultdict(list)
    for row in rows:
        overlays[(row["family"], row["setting_id"], row["trait_id"], row["interval"])].append(row)
    overlay_images = []
    for key, subset in sorted(overlays.items()):
        by_run = defaultdict(list)
        for row in subset:
            by_run[row["run_id"]].append(row)
        if len(by_run) < 2:
            continue
        binary = key[0] in ("tabular_binary", "llm_binary")
        coeff, state = ("S", "T") if binary else ("beta", "mu")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
        for rid, series in sorted(by_run.items()):
            for ax, x in zip(axes, [state, "step"]):
                pairs = _pairs(series, x, coeff)
                if pairs:
                    a = np.asarray(pairs)
                    ax.scatter(a[:, 0], a[:, 1], s=8, alpha=.55, label=rid)
                ax.set(xlabel=x, ylabel=coeff)
        axes[1].legend(fontsize=6, loc="best")
        title = f"{key[1]} · trait {key[2]} · gap {key[3]} · individual runs, no pooling"
        fig.suptitle(title)
        fig.tight_layout()
        overlay_images.append((title, _save(fig, folder, _slug("overlay-" + "-".join(map(str, key))))))
    write_csv(output / "series_summary.csv", summary)
    lines = ["# Closure-study diagnostic report", "",
             "Build a small dynamical model of unrewarded trait change by starting from the Price equation, identifying what remains unclosed, and using experiments to test closure assumptions and estimate their constants.", "",
             "Both binary and continuous traits are central. Seek reusable closure forms with setting-specific constants. Evaluate the resulting models through their interpretation, measured closure residuals, and generated trajectories. Incremental forecasting gains from Price measurements are a secondary question.", "",
             "## Milestone status", "",
             f"Recovered {len(rows)} transition/trait records covering {len(grouped)} trait series and {len({r['run_id'] for r in rows})} recorded runs. Run identities do not certify independent training replicates.",
             "No closure has been fitted or selected. No coefficients are pooled. No training or GPU reevaluation was launched. Stop here for joint diagnostic review.", "",
             "## Read the evidence with its measurement limits", "",
             "- [Accounting audit](accounting_audit.md) records missing measurements and unresolved historical sampling/scoring questions.",
             "- [Run inventory](run_inventory.csv), [study table](transitions.csv), [conditional bin table](conditional_bins.csv), and [accounting table](accounting.csv) retain full precision and provenance labels.",
             "- Exact-table quantities and historical sampled LLM estimates have different uncertainty. Missing LLM error bars indicate missing information, not zero uncertainty.",
             "- S and beta are measured transition ratios, not fitted parameters. Output S is sparse because old-state prevalence was not logged every update.",
             "- Per-run plots list their transition gaps. Overlays separate gaps and settings; no coefficient is divided by interval length to invent a per-update law.",
             "- Conditional plots use weighted-quantile bins at data-independent early/middle/late transitions. Inspect mean-omega normalization alongside conditional residuals.",
             "- Q − beta M3 tests an additional reweighting approximation; Delta mu = beta V alone does not test it. Constant skewness would not establish preservation of all distributional shape.",
             "- Shared-state overlays are descriptive. Similar monotonic trajectories can also follow a reproducible time schedule.", "",
             "## Questions for discussion", "",
             "1. Which individual runs are consistent with approximately constant selection, and which show structured state/time dependence?",
             "2. Are differences across runs substantive or unresolved by current measurement precision?",
             "3. Is conditional linear reweighting adequate, judging higher-moment residuals as well as binned ratios?",
             "4. Which missing measurements must be recovered before choosing any closure?", "",
             "## Cross-run overlays", ""]
    lines.extend(f"- [{title}]({path})" for title, path in overlay_images)
    lines.extend(["", "## Individual runs and features", ""])
    for title, paths in figures:
        lines.append(f"- {title}: " + " · ".join(f"[{('state diagnostics' if i == 0 else 'conditional reweighting')}]({p})" for i, p in enumerate(paths)))
    with (output / "report.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    html = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Closure diagnostics</title>",
            "<style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px system-ui;color:#17212b}img{max-width:100%}summary{cursor:pointer;padding:.7rem}details{border-bottom:1px solid #ddd}a{color:#145a86}p{line-height:1.5}</style>",
            "<h1>GRPO trait closure: diagnostic milestone</h1>",
            "<p>Individual runs first. No pooled coefficients, fitted closures, or model selection.</p>",
            "<p>Historical LLM estimates lack the response and panel information needed for calibrated uncertainty. Exact-table and sampled diagnostics are labelled separately.</p>",
            "<p><a href='report.md'>Report and interpretation limits</a> · <a href='accounting_audit.md'>Accounting audit</a> · <a href='transitions.csv'>Study table</a> · <a href='run_inventory.csv'>Inventory</a></p>",
            "<h2>Cross-run overlays</h2>"]
    for title, path in overlay_images:
        html.append(f"<details><summary>{escape(title)}</summary><img loading='lazy' src='{escape(path)}' alt='{escape(title)}'></details>")
    html.append("<h2>Individual runs and traits</h2>")
    for title, paths in figures:
        html.append(f"<details><summary>{escape(title)}</summary>" + "".join(f"<img loading='lazy' src='{escape(p)}' alt='{escape(title)}'>" for p in paths) + "</details>")
    html.append("</html>")
    with (output / "index.html").open("x") as handle:
        handle.write("\n".join(html))
    artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "artifact_manifest.json", {"schema_version": 1, "files": artifacts})
    print(f"Report: {output / 'index.html'}", flush=True)
