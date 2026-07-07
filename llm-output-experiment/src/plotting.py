from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

_MARKERS = ("o", "s", "^", "D")


def load_metrics(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _carry_series(metrics: list[dict], getter, default=float("nan")):
    values, last = [], default
    for item in metrics:
        try:
            last = getter(item)
        except (KeyError, TypeError):
            pass
        values.append(last)
    return values


def _first_real_value(values, default=0.0):
    for value in values:
        try:
            if value == value:  # not NaN
                return value
        except TypeError:
            return value
    return default


def _save(fig, path_stem: Path) -> list[Path]:
    written = []
    for ext in ("png", "pdf"):
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150)
        written.append(p)
    plt.close(fig)
    return written


def _line_plot(path_stem: Path, x, series, ylabel: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6, 4))
    for (label, y), marker in zip(series, _MARKERS):
        ax.plot(x, y, marker=marker, markersize=3, label=label)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path_stem)


def _price_plot(path_stem: Path, metrics: list[dict], distribution: str,
                trait_name: str, observed_key: str) -> list[Path]:
    x = [m["step"] for m in metrics]
    observed = _carry_series(metrics, lambda m: m["observed_eval"][distribution][observed_key])
    predicted = _carry_series(metrics, lambda m: m["price"][distribution][trait_name]["cov_cum"])
    if observed:
        base = _first_real_value(observed)
        observed = [v - base for v in observed]
    residual = [o - p for o, p in zip(observed, predicted)]
    return _line_plot(
        path_stem, x,
        [("Observed: T_t - T_0", observed),
         ("Predicted: cumulative Cov(omega, s)", predicted),
         ("Residual", residual)],
        "Cumulative trait change",
    )


def _omega_plot(path_stem: Path, x, mean_omega, ess) -> list[Path]:
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    a1.plot(x, mean_omega, color=colors[0], marker="o", markersize=3)
    a1.axhline(1.0, ls="--", lw=0.8, color="grey")
    a1.set_ylabel(r"$\bar\omega$")
    a2.plot(x, ess, color=colors[2], marker="o", markersize=3)
    a2.set_ylabel("ESS")
    a2.set_xlabel("GRPO step")
    a2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save(fig, path_stem)


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    metrics = load_metrics(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return []

    x = [m["step"] for m in metrics]
    written: list[Path] = []

    written += _line_plot(
        plots_dir / "reward_accuracy", x,
        [("Reward", _carry_series(metrics, lambda m: m["train"]["reward_mean"])),
         ("Accuracy", _carry_series(metrics, lambda m: m["train"]["accuracy"]))],
        "Level",
    )

    written += _omega_plot(
        plots_dir / "omega_diagnostics", x,
        _carry_series(metrics, lambda m: m["price"]["eval_wrong_hint"]["output_agreement"]["mean_omega"]),
        _carry_series(metrics, lambda m: m["price"]["eval_wrong_hint"]["output_agreement"]["ess"]),
    )

    for distribution in ("eval_wrong_hint", "eval_balanced_hint"):
        written += _price_plot(
            plots_dir / f"output_agreement_price_check_{distribution}",
            metrics, distribution, "output_agreement", "agreement_rate",
        )

    written += _price_plot(
        plots_dir / "activation_agreement_price_check_eval_wrong_hint",
        metrics, "eval_wrong_hint", "activation_agreement", "mean_activation_agreement_score",
    )
    return written
