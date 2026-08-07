from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _series(metrics: list[dict], getter, default=float("nan")):
    values = []
    for item in metrics:
        try:
            values.append(getter(item))
        except (KeyError, TypeError):
            values.append(default)
    return values


def _carry_series(metrics: list[dict], getter, default=float("nan")):
    values = []
    last = default
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
            if not math.isnan(value):
                return value
        except TypeError:
            return value
    return default


def _summary_values(runs: list[dict], key: str):
    return [float("nan") if item.get(key) is None else item[key] for item in runs]


def _save_line_plot(path: Path, x, series, ylabel: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, y in series:
        ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _price_plot(path: Path, metrics: list[dict], distribution: str, trait_name: str, observed_key: str):
    x = [m["step"] for m in metrics]
    observed = _carry_series(metrics, lambda m: m["observed_eval"][distribution][observed_key])
    predicted = _carry_series(metrics, lambda m: m["price"][distribution][trait_name]["cov_cum"])
    if observed:
        base = _first_real_value(observed)
        observed = [value - base for value in observed]
    residual = [obs - pred for obs, pred in zip(observed, predicted)]
    _save_line_plot(
        path,
        x,
        [("Observed: T_t - T_0", observed), ("Predicted: cumulative Cov(omega, s)", predicted), ("Residual", residual)],
        "Cumulative trait change",
    )


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    metrics = load_metrics(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return []

    x = [m["step"] for m in metrics]
    written = []

    targets = [
        (
            "reward_accuracy.png",
            [
                ("Reward", _carry_series(metrics, lambda m: m["train"]["reward_mean"])),
                ("Accuracy", _carry_series(metrics, lambda m: m["train"]["accuracy"])),
            ],
            "Level",
        ),
        (
            "wrong_hint_sycophancy.png",
            [
                ("Wrong-hint agreement", _carry_series(metrics, lambda m: m["observed_eval"]["eval_wrong_hint"]["wrong_hint_agreement_rate"])),
                ("Sycophantic error", _carry_series(metrics, lambda m: m["observed_eval"]["eval_wrong_hint"]["sycophantic_error_rate"])),
                ("Correct disagreement", _carry_series(metrics, lambda m: m["observed_eval"]["eval_wrong_hint"]["correct_disagreement_rate"])),
            ],
            "Rate",
        ),
        (
            "activation_invariance.png",
            [
                ("Max abs", _carry_series(metrics, lambda m: m["activation_invariance"]["max_abs"])),
                ("Mean abs", _carry_series(metrics, lambda m: m["activation_invariance"]["mean_abs"])),
            ],
            "Activation score change",
        ),
        (
            "omega_diagnostics.png",
            [
                ("mean omega", _carry_series(metrics, lambda m: m["price"]["eval_wrong_hint"]["output_agreement"]["mean_omega"])),
                ("std omega", _carry_series(metrics, lambda m: m["price"]["eval_wrong_hint"]["output_agreement"]["std_omega"])),
                ("ESS", _carry_series(metrics, lambda m: m["price"]["eval_wrong_hint"]["output_agreement"]["ess"])),
            ],
            "Diagnostic",
        ),
        (
            "length_control.png",
            [
                ("Train length", _carry_series(metrics, lambda m: m["train"]["mean_completion_token_length"])),
                ("Wrong-hint eval length", _carry_series(metrics, lambda m: m["observed_eval"]["eval_wrong_hint"]["mean_completion_token_length"])),
            ],
            "Tokens",
        ),
    ]
    for filename, series, ylabel in targets:
        path = plots_dir / filename
        _save_line_plot(path, x, series, ylabel)
        written.append(path)

    for distribution in ("eval_wrong_hint", "eval_balanced_hint"):
        path = plots_dir / f"output_agreement_price_check_{distribution}.png"
        _price_plot(path, metrics, distribution, "output_agreement", "agreement_rate")
        written.append(path)

    path = plots_dir / "activation_agreement_price_check_eval_wrong_hint.png"
    _price_plot(path, metrics, "eval_wrong_hint", "activation_agreement", "mean_activation_agreement_score")
    written.append(path)
    return written


def plot_reliability_sweep(sweep_dir: str | Path) -> list[Path]:
    sweep_dir = Path(sweep_dir)
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text())
    if not summary.get("runs"):
        return []
    runs = summary.get("runs", [])
    reliabilities = [item["train_hint_correct_probability"] for item in runs]
    agreement = _summary_values(runs, "final_wrong_hint_agreement_rate")
    sycophancy = _summary_values(runs, "final_sycophantic_error_rate")
    price = _summary_values(runs, "final_output_agreement_price_cum")

    written = []
    for filename, values, ylabel in [
        ("reliability_sweep_agreement.png", agreement, "Wrong-hint agreement"),
        ("reliability_sweep_wrong_hint_sycophancy.png", sycophancy, "Sycophantic error"),
        ("reliability_sweep_price_summary.png", price, "Cumulative Price estimate"),
    ]:
        path = plots_dir / filename
        _save_line_plot(path, reliabilities, [(ylabel, values)], ylabel)
        written.append(path)
    return written
