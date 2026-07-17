from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


EVAL_STEPS = list(range(0, 101, 5))
THRESHOLDS = {
    "final_agreement_min": 0.95,
    "final_agreement_abs_diff": 0.05,
    "final_accuracy_abs_diff": 0.05,
    "final_invalid_abs_diff": 0.05,
    "final_no_hint_accuracy_abs_diff": 0.05,
    "final_no_hint_invalid_abs_diff": 0.05,
    "final_observed_drift_abs_diff": 0.15,
    "agreement_trajectory_rmse": 0.15,
    "threshold_crossing_step_abs_diff": 10,
    "final_price_abs_diff": 0.25,
    "price_trajectory_rmse": 0.25,
    "final_price_residual": 0.30,
}


def load_metrics(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing metrics file: {path}")
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if [row.get("step") for row in rows] != list(range(101)):
        raise ValueError(f"{path} must contain exactly steps 0 through 100")
    return rows


def _observed_series(
    rows: list[dict],
    key: str,
    distribution: str = "eval_wrong_hint",
) -> list[float]:
    by_step = {
        row["step"]: row.get("observed_eval", {}).get(distribution, {})
        for row in rows
        if row.get("observed_eval")
    }
    missing = [step for step in EVAL_STEPS if key not in by_step.get(step, {})]
    if missing:
        raise ValueError(f"missing {distribution}/{key} at steps {missing}")
    return [float(by_step[step][key]) for step in EVAL_STEPS]


def _price_series(rows: list[dict], trait: str = "output_agreement") -> list[float]:
    by_step = {0: 0.0}
    for row in rows:
        block = row.get("price", {}).get("eval_wrong_hint", {}).get(trait, {})
        if "cov_cum" in block:
            by_step[row["step"]] = float(block["cov_cum"])
    missing = [step for step in EVAL_STEPS if step not in by_step]
    if missing:
        raise ValueError(f"missing cumulative {trait} Price estimate at steps {missing}")
    return [by_step[step] for step in EVAL_STEPS]


def _rmse(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("RMSE inputs must be non-empty and equal length")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _first_crossing(values: list[float], threshold: float = 0.9) -> int | None:
    for step, value in zip(EVAL_STEPS, values):
        if value >= threshold:
            return step
    return None


def compare_metrics(reference_rows: list[dict], candidate_rows: list[dict], label: str) -> dict:
    ref_agreement = _observed_series(reference_rows, "agreement_rate")
    cand_agreement = _observed_series(candidate_rows, "agreement_rate")
    ref_accuracy = _observed_series(reference_rows, "accuracy")
    cand_accuracy = _observed_series(candidate_rows, "accuracy")
    ref_invalid = _observed_series(reference_rows, "invalid_output_rate")
    cand_invalid = _observed_series(candidate_rows, "invalid_output_rate")
    ref_no_hint_accuracy = _observed_series(reference_rows, "accuracy", "eval_no_hint")
    cand_no_hint_accuracy = _observed_series(candidate_rows, "accuracy", "eval_no_hint")
    ref_no_hint_invalid = _observed_series(reference_rows, "invalid_output_rate", "eval_no_hint")
    cand_no_hint_invalid = _observed_series(candidate_rows, "invalid_output_rate", "eval_no_hint")
    ref_drift = _observed_series(reference_rows, "output_agreement_observed_drift")
    cand_drift = _observed_series(candidate_rows, "output_agreement_observed_drift")
    ref_price = _price_series(reference_rows)
    cand_price = _price_series(candidate_rows)
    ref_shuffled_price = _price_series(reference_rows, "output_agreement_shuffled")
    cand_shuffled_price = _price_series(candidate_rows, "output_agreement_shuffled")
    ref_crossing = _first_crossing(ref_agreement)
    cand_crossing = _first_crossing(cand_agreement)
    crossing_diff = None if ref_crossing is None or cand_crossing is None else abs(cand_crossing - ref_crossing)

    comparisons = {
        "final_agreement_abs_diff": abs(cand_agreement[-1] - ref_agreement[-1]),
        "final_accuracy_abs_diff": abs(cand_accuracy[-1] - ref_accuracy[-1]),
        "final_invalid_abs_diff": abs(cand_invalid[-1] - ref_invalid[-1]),
        "final_no_hint_accuracy_abs_diff": abs(cand_no_hint_accuracy[-1] - ref_no_hint_accuracy[-1]),
        "final_no_hint_invalid_abs_diff": abs(cand_no_hint_invalid[-1] - ref_no_hint_invalid[-1]),
        "no_hint_accuracy_trajectory_rmse": _rmse(ref_no_hint_accuracy, cand_no_hint_accuracy),
        "no_hint_invalid_trajectory_rmse": _rmse(ref_no_hint_invalid, cand_no_hint_invalid),
        "final_observed_drift_abs_diff": abs(cand_drift[-1] - ref_drift[-1]),
        "agreement_trajectory_rmse": _rmse(ref_agreement, cand_agreement),
        "threshold_crossing_step_abs_diff": crossing_diff,
        "final_price_abs_diff": abs(cand_price[-1] - ref_price[-1]),
        "price_trajectory_rmse": _rmse(ref_price, cand_price),
        "final_shuffled_price_abs_diff": abs(cand_shuffled_price[-1] - ref_shuffled_price[-1]),
        "shuffled_price_trajectory_rmse": _rmse(ref_shuffled_price, cand_shuffled_price),
        "candidate_final_price_residual": abs(cand_price[-1] - cand_drift[-1]),
    }
    gates = {
        "candidate_final_agreement": cand_agreement[-1] >= THRESHOLDS["final_agreement_min"],
        "final_agreement_similarity": comparisons["final_agreement_abs_diff"] <= THRESHOLDS["final_agreement_abs_diff"],
        "final_accuracy_similarity": comparisons["final_accuracy_abs_diff"] <= THRESHOLDS["final_accuracy_abs_diff"],
        "final_invalid_similarity": comparisons["final_invalid_abs_diff"] <= THRESHOLDS["final_invalid_abs_diff"],
        "final_no_hint_accuracy_similarity": comparisons["final_no_hint_accuracy_abs_diff"]
        <= THRESHOLDS["final_no_hint_accuracy_abs_diff"],
        "final_no_hint_invalid_similarity": comparisons["final_no_hint_invalid_abs_diff"]
        <= THRESHOLDS["final_no_hint_invalid_abs_diff"],
        "final_observed_drift_similarity": comparisons["final_observed_drift_abs_diff"] <= THRESHOLDS["final_observed_drift_abs_diff"],
        "agreement_trajectory_similarity": comparisons["agreement_trajectory_rmse"] <= THRESHOLDS["agreement_trajectory_rmse"],
        "threshold_crossing_similarity": crossing_diff is not None
        and crossing_diff <= THRESHOLDS["threshold_crossing_step_abs_diff"],
        "final_price_similarity": comparisons["final_price_abs_diff"] <= THRESHOLDS["final_price_abs_diff"],
        "price_trajectory_similarity": comparisons["price_trajectory_rmse"] <= THRESHOLDS["price_trajectory_rmse"],
        "candidate_price_positive": cand_price[-1] > 0.0,
        "candidate_price_tracks_observed": comparisons["candidate_final_price_residual"] <= THRESHOLDS["final_price_residual"],
    }
    return {
        "label": label,
        "passed": all(gates.values()),
        "thresholds": THRESHOLDS,
        "gates": gates,
        "comparisons": comparisons,
        "eval_steps": EVAL_STEPS,
        "reference": {
            "final_agreement": ref_agreement[-1],
            "final_accuracy": ref_accuracy[-1],
            "final_invalid_output_rate": ref_invalid[-1],
            "final_no_hint_accuracy": ref_no_hint_accuracy[-1],
            "final_no_hint_invalid_output_rate": ref_no_hint_invalid[-1],
            "final_observed_drift": ref_drift[-1],
            "first_agreement_0_9_step": ref_crossing,
            "final_price_cumulative": ref_price[-1],
            "final_shuffled_price_cumulative": ref_shuffled_price[-1],
            "agreement_trajectory": ref_agreement,
            "price_trajectory": ref_price,
            "no_hint_accuracy_trajectory": ref_no_hint_accuracy,
            "no_hint_invalid_trajectory": ref_no_hint_invalid,
            "shuffled_price_trajectory": ref_shuffled_price,
        },
        "candidate": {
            "final_agreement": cand_agreement[-1],
            "final_accuracy": cand_accuracy[-1],
            "final_invalid_output_rate": cand_invalid[-1],
            "final_no_hint_accuracy": cand_no_hint_accuracy[-1],
            "final_no_hint_invalid_output_rate": cand_no_hint_invalid[-1],
            "final_observed_drift": cand_drift[-1],
            "first_agreement_0_9_step": cand_crossing,
            "final_price_cumulative": cand_price[-1],
            "final_shuffled_price_cumulative": cand_shuffled_price[-1],
            "agreement_trajectory": cand_agreement,
            "price_trajectory": cand_price,
            "no_hint_accuracy_trajectory": cand_no_hint_accuracy,
            "no_hint_invalid_trajectory": cand_no_hint_invalid,
            "shuffled_price_trajectory": cand_shuffled_price,
        },
    }


def _write_comparison(report: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = report["label"]
    json_path = output_dir / f"{label}_comparison.json"
    md_path = output_dir / f"{label}_comparison.md"
    agreement_path = output_dir / f"{label}_agreement_overlay.png"
    price_path = output_dir / f"{label}_price_overlay.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# {label.replace('_', ' ').title()} Comparison",
        "",
        f"Overall gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Gate | Result |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in report["gates"].items())
    lines.extend(["", "## Numeric comparisons", "", "| Metric | Value |", "| --- | ---: |"]) 
    for name, value in report["comparisons"].items():
        rendered = "None" if value is None else f"{value:.6f}"
        lines.append(f"| {name} | {rendered} |")
    md_path.write_text("\n".join(lines) + "\n")

    for path, key, ylabel in [
        (agreement_path, "agreement_trajectory", "Wrong-hint agreement"),
        (price_path, "price_trajectory", "Cumulative output Price"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(EVAL_STEPS, report["reference"][key], marker="o", label="Reference")
        ax.plot(EVAL_STEPS, report["candidate"][key], marker="o", label="Candidate")
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return [json_path, md_path, agreement_path, price_path]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(reference_dir: str | Path) -> dict:
    reference_dir = Path(reference_dir)
    files = []
    for path in sorted(item for item in reference_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(reference_dir)),
                "size": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    if not files:
        raise ValueError(f"reference directory contains no files: {reference_dir}")
    return {"reference_dir": str(reference_dir), "files": files}


def write_manifest(reference_dir: str | Path, manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(build_manifest(reference_dir), indent=2) + "\n")
    return manifest_path


def verify_manifest(reference_dir: str | Path, manifest_path: str | Path) -> bool:
    expected = json.loads(Path(manifest_path).read_text())
    return expected["files"] == build_manifest(reference_dir)["files"]


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--reference-dir", required=True)
    snapshot.add_argument("--manifest", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--reference-dir", required=True)
    verify.add_argument("--manifest", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference-dir", required=True)
    compare.add_argument("--candidate-dir", required=True)
    compare.add_argument("--label", required=True)
    compare.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "snapshot":
        print(write_manifest(args.reference_dir, args.manifest))
        return 0
    if args.command == "verify":
        passed = verify_manifest(args.reference_dir, args.manifest)
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1

    report = compare_metrics(load_metrics(args.reference_dir), load_metrics(args.candidate_dir), args.label)
    for path in _write_comparison(report, Path(args.output_dir)):
        print(path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
