"""Offline commands stop at diagnostics. No model selection or training entrypoint."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .adapters import LOADERS
from .io import inventory_entry, jsonable, load_registry, read_jsonl, sha256, write_csv, write_json


def write_jsonl(path, rows):
    with Path(path).open("x") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), allow_nan=False) + "\n")


def inventory(registry, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = load_registry(registry)
    records = []
    for entry in entries:
        record = inventory_entry(entry)
        if entry["family"] in ("tabular_binary", "neural_continuous") and Path(entry["trajectories_path"]).is_file():
            with np.load(entry["trajectories_path"], allow_pickle=False) as archive:
                for seed in archive["seeds"]:
                    records.append({**record, "source_run_id": entry["run_id"],
                                    "run_id": f"{entry['run_id']}:seed-{int(seed)}", "archive_seed": int(seed)})
        else:
            records.append({**record, "source_run_id": entry["run_id"]})
    write_json(output / "inventory.json", records)
    write_csv(output / "run_inventory.csv", records)
    source_root = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    write_json(output / "manifest.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(Path(registry).resolve()), "registry_sha256": sha256(registry),
        "analysis_base_revision": revision,
        "analysis_source_hashes": {str(p.relative_to(source_root)): sha256(p) for p in sorted(source_root.rglob("*.py"))},
        "python": platform.python_version(), "numpy": np.__version__,
        "phase": "per_run_diagnostics_only", "source_inventory": "inventory.json",
        "historical_data_policy": "explicit development entries only; no untouched suffix discovery",
    })
    print(f"Inventoried {len(records)} recorded runs from {len(entries)} sources", flush=True)
    return records


def diagnose(registry, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, bins, accounting, issues = [], [], [], []
    for entry in load_registry(registry):
        rid = entry["run_id"]
        required = {"tabular_binary": "trajectories_path", "neural_continuous": "trajectories_path",
                    "llm_binary": "metrics_path", "llm_continuous": "pool_path"}[entry["family"]]
        if not Path(entry[required]).is_file():
            issues.append({"run_id": rid, "severity": "unavailable", "issue": f"missing {required}: {entry[required]}"})
            continue
        try:
            result, conditional, checks = LOADERS[entry["family"]](entry)
        except (ValueError, KeyError, OSError, FloatingPointError) as error:
            issues.append({"run_id": rid, "severity": "excluded", "issue": f"{type(error).__name__}: {error}"})
            print(f"Excluded {rid}: {error}", flush=True)
            continue
        rows.extend(result)
        bins.extend(conditional)
        accounting.extend(checks)
        print(f"{rid}: {len(result)} transition/trait records", flush=True)
    for name, values in [("transitions", rows), ("conditional_bins", bins), ("accounting", accounting), ("source_issues", issues)]:
        write_jsonl(output / f"{name}.jsonl", values)
        write_csv(output / f"{name}.csv", values)
    print(f"Exported {len(rows)} records; {len(issues)} unavailable/excluded sources", flush=True)


def audit(output):
    output = Path(output)
    rows = read_jsonl(output / "transitions.jsonl")
    issues = read_jsonl(output / "source_issues.jsonl")
    text = ["# Accounting and measurement audit", "",
            "This is an offline diagnostic audit, not certification of historical LLM sampling/scoring.", "",
            "- Exact tables: inspect mean/variance identity residuals and float32 archive reconstruction residuals.",
            "- Output LLM: covariance is logged every update; S uses only recorded independent old-checkpoint prevalence. No states are interpolated.",
            "- Output accounting across sparse evaluations sums unit covariances. Those sums are not aggregate likelihood ratios or aggregate S measurements.",
            "- SAE pools: C and Q are recomputed at full precision from raw scores and unnormalized likelihood ratios. Both use the same old-policy panel.",
            "- Separate SAE direct means were generated independently of the saved Price pool where the corresponding generator/mean archive exists. Only their aggregate means remain.",
            "- Historical rounded direct drifts have weaker panel provenance and are labelled separately.",
            "- LLM sampling intervals are unavailable here: prompt IDs, direct response samples, and their joint sampling covariance were not retained. ESS is a weight diagnostic, not an uncertainty estimate.",
            "- No pooled run estimates, confidence bands from feature replicates, or tests of coefficient constancy are performed.",
            "- A zero/very small variance can make beta or S unidentifiable/noisy. Numerical zeros are missing; statistical precision of nonzero estimates remains unassessed.",
            "- Binary Q is algebraically reconstructed; its relationship to M3 is not an independent test of conditional linearity.",
            "", "## Descriptive checks", "",
            "| Family | Rows | S present | beta present | Max absolute accounting residual |", "|---|---:|---:|---:|---:|"]
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        residuals = [abs(r["accounting_residual"]) for r in subset if r.get("accounting_residual") is not None]
        maximum = f"{max(residuals):.6g}" if residuals else "unavailable per unit transition"
        text.append(f"| {family} | {len(subset)} | {sum(r.get('S') is not None for r in subset)} | {sum(r.get('beta') is not None for r in subset)} | {maximum} |")
    text.extend(["", "These maxima describe existing point estimates; they do not establish systematic bias or closure failure.",
                 "", "## Measurement gaps for discussion", "",
                 "1. Recheck historical LLM sampling/scoring parity (including EOS, generation transforms, precision, and action masks) at the actual source revision.",
                 "2. Recover or reevaluate matched LLM scores with prompt/panel IDs and independent direct samples before precision-dependent mechanistic conclusions.",
                 "3. Verify RNG provenance before treating named LLM seeds as independent training replicates.",
                 "4. The last neural transition lacks next-state particle weights: its stored mean increment is retained, but Q and next variance stay missing.",
                 "5. Investigate support failures or material archived-versus-reconstructed discrepancies before interpreting affected exact-table ratios.",
                 "", "## Source exclusions", ""])
    text.extend([f"- {i['run_id']}: {i['issue']}" for i in issues] or ["None."])
    with (output / "accounting_audit.md").open("x") as handle:
        handle.write("\n".join(text) + "\n")


def run(registry, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; historical outputs are never overwritten")
    inventory(registry, output)
    diagnose(registry, output)
    audit(output)
    from .report import report
    report(output)
