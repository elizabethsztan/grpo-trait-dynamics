"""Explicit artifact selection and lossless, versioned diagnostic exports."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from . import SCHEMA_VERSION


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    with Path(path).open("x") as handle:
        json.dump(jsonable(data), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    preferred = ["run_id", "family", "setting_id", "trait_id", "step", "step_end", "interval"]
    keys = set().union(*(r.keys() for r in rows)) if rows else set(preferred)
    fields = [k for k in preferred if k in keys] + sorted(keys - set(preferred))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = jsonable(row)
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in clean.items()})


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def steps(values):
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("checkpoint steps must be a finite vector")
    if (raw < 0).any() or (raw != np.floor(raw)).any() or (np.diff(raw) <= 0).any():
        raise ValueError("checkpoint steps must be increasing nonnegative integers")
    return raw.astype(int)


def load_registry(path):
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported study registry schema")
    root = (path.parent / document.get("root", ".")).resolve()
    seen = set()
    runs = []
    for entry in document["runs"]:
        entry = dict(entry)
        if entry["run_id"] in seen:
            raise ValueError(f"duplicate run_id: {entry['run_id']}")
        seen.add(entry["run_id"])
        if entry.get("inspection_status") != "development":
            raise ValueError(f"diagnostic registry must not open non-development run {entry['run_id']}")
        for key, value in list(entry.items()):
            if key.endswith("_path"):
                entry[key] = str((root / value).resolve())
        runs.append(entry)
    return runs


def inventory_entry(entry):
    paths = {k: v for k, v in entry.items() if k.endswith("_path")}
    present = {k: v for k, v in paths.items() if Path(v).is_file()}
    config = yaml.safe_load(Path(entry["config_path"]).read_text()) if "config_path" in present else {}
    train_key = {"neural_continuous": "NeuralTrainConfig", "llm_continuous": "GRPOConfig"}.get(
        entry["family"], "TrainConfig")
    train = config.get(train_key, {})
    checkpoint_dir = Path(entry.get("pool_path", entry.get("metrics_path", ""))).parent / "checkpoints"
    return {**entry, "schema_version": SCHEMA_VERSION, "paths": paths,
            "source_hashes": {k: sha256(v) for k, v in present.items()},
            "missing_paths": sorted(set(paths) - set(present)), "configuration": config,
            "learning_rate": train.get("learning_rate", train.get("lr")),
            "eta": train.get("eta") if entry["family"] == "tabular_binary" else None,
            "generation": config.get("GenConfig", config.get("GenerationConfig", {})),
            "model": config.get("ModelConfig", {}),
            "layer_scope": config.get("LoRAConfig", {}).get("layer_scope", train.get("lora_layers")),
            "recorded_seeds": {k: v.get("seed") for k, v in config.items() if isinstance(v, dict) and "seed" in v},
            "checkpoint_count": len(list(checkpoint_dir.glob("step_*"))) if checkpoint_dir.is_dir() else 0,
            "training_revision_status": "unverified_unless_source_manifest_establishes_it",
            "randomness_status": entry.get("randomness_status", "training_replicate_independence_not_verified")}
