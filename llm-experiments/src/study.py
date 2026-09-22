"""Portable manifests, matched evaluation banks, and study artifact I/O."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import tempfile

from .data import LETTERS, MCArithmeticExample, generate_examples
from .prompts import render_prompt

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTIONS = ("eval_wrong_hint", "eval_correct_hint", "eval_balanced_hint", "eval_no_hint")
PRICE_DISTRIBUTIONS = ("eval_wrong_hint", "eval_balanced_hint")
RELIABILITIES = (0.10, 0.25, 0.50, 0.75, 0.90)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, update=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not update:
        with path.open("x") as stream:
            stream.write(text)
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".status-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def write_row(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def artifact(root, path, **metadata):
    path = Path(path)
    return {"path": str(path.relative_to(root)), "sha256": digest(path), **metadata}


def verify_artifact(root, reference):
    root = Path(root).resolve()
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root) or digest(path) != reference["sha256"]:
        raise ValueError(f"artifact hash/path mismatch: {reference['path']}")
    return path


def stream_seed(seed, *labels):
    material = json.dumps([seed, *labels], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=EXPERIMENT_ROOT, text=True).strip()

    source_paths = sorted((EXPERIMENT_ROOT / "src").glob("*.py")) + [
        EXPERIMENT_ROOT / "run_paper_study.py", EXPERIMENT_ROOT / "train_grpo_price.py",
    ]
    versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "source_status": git("status", "--porcelain", "--", "src", "run_paper_study.py", "train_grpo_price.py"),
        "source_sha256": {str(p.relative_to(EXPERIMENT_ROOT)): digest(p) for p in source_paths},
        "python": platform.python_version(), "packages": versions,
    }


def make_bank(config, seed):
    n = config["PaperStudyConfig"]["bank_capacity"]
    data = config["DataConfig"]
    examples = generate_examples(n, seed, "paper_bank", difficulty=data["difficulty"], has_hint=False)
    rng = random.Random(stream_seed(seed, "hint_variants"))
    # Every even-sized prefix is balanced, allowing nested 512/1024/2048 banks.
    balanced_correct = {start + rng.randrange(2) for start in range(0, n, 2)}
    rows = []
    for group, example in enumerate(examples):
        wrong = rng.choice([letter for letter in LETTERS if letter != example.gold_choice])
        phrase = rng.choice(data["eval_hint_phrases"])
        variants = {}
        for distribution in DISTRIBUTIONS:
            hint = {
                "eval_wrong_hint": wrong, "eval_correct_hint": example.gold_choice,
                "eval_balanced_hint": example.gold_choice if group in balanced_correct else wrong,
                "eval_no_hint": None,
            }[distribution]
            variant = replace(
                example, split=distribution, user_hint=hint,
                hint_is_correct=None if hint is None else hint == example.gold_choice,
                hint_phrase=None if hint is None else phrase,
                prompt_text=render_prompt(example.problem_text, example.options, hint, phrase),
            )
            variants[distribution] = variant.to_json_dict()
        rows.append({"group_id": example.problem_id, "variants": variants})
    return rows


def training_examples(config, step):
    """Match problems and hint draws across conditions within each training seed."""
    seed, data = config["RunConfig"]["seed"], config["DataConfig"]
    examples = generate_examples(
        config["TrainConfig"]["train_prompts_per_step"],
        stream_seed(seed, "train_problems", step), f"train_step{step:04d}",
        difficulty=data["difficulty"], has_hint=False,
    )
    correctness = random.Random(stream_seed(seed, "train_hint_correctness", step))
    wrong_choices = random.Random(stream_seed(seed, "train_wrong_hint", step))
    phrases = random.Random(stream_seed(seed, "train_hint_phrase", step))
    probability = float(data["train_hint_correct_probability"])
    if not 0 <= probability <= 1:
        raise ValueError("train_hint_correct_probability must be in [0, 1]")
    matched = []
    for example in examples:
        draw = correctness.random()
        wrong = wrong_choices.choice([letter for letter in LETTERS if letter != example.gold_choice])
        phrase = phrases.choice(data["train_hint_phrases"])
        hint = (example.gold_choice if draw < probability else wrong) if data["train_has_hint"] else None
        matched.append(replace(
            example, user_hint=hint, hint_is_correct=None if hint is None else hint == example.gold_choice,
            hint_phrase=None if hint is None else phrase,
            prompt_text=render_prompt(example.problem_text, example.options, hint, phrase),
        ))
    return matched


def validate_config(config):
    model = config["ModelConfig"]
    if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
        raise ValueError("paper study requires an exact model/tokenizer revision")
    if config["ActivationProbeConfig"]["enabled"] or config["LoRAConfig"].get("layer_scope") != "all":
        raise ValueError("paper study requires all-layer LoRA and no activation probe")
    if config["TrainConfig"].get("kl_coef", 0) != 0:
        raise ValueError("the simplified training objective does not implement a KL penalty")
    for section, keys in {
        "TrainConfig": ("num_steps", "train_prompts_per_step", "group_size"),
        "PriceConfig": ("prompts_per_distribution", "completions_per_prompt"),
        "ObservedEvalConfig": ("prompts_per_distribution", "completions_per_prompt", "eval_every"),
    }.items():
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    n = config["PriceConfig"]["prompts_per_distribution"]
    if n % 2 or n != config["ObservedEvalConfig"]["prompts_per_distribution"]:
        raise ValueError("Price and observed evaluation must use the same even-sized bank")
    capacity = config["PaperStudyConfig"]["bank_capacity"]
    if type(capacity) is not int or capacity < n or capacity % 2:
        raise ValueError("bank capacity must be even and at least the measurement prompt count")
    if set(config["PriceConfig"]["eval_distributions"]) != set(PRICE_DISTRIBUTIONS):
        raise ValueError("Price distributions must be wrong-hint and balanced-hint")


def prepare_study(config, output, cohort):
    validate_config(config)
    if cohort not in ("pilot", "paper"):
        raise ValueError("cohort must be pilot or paper")
    settings = config["PaperStudyConfig"]
    if cohort == "paper":
        seeds = settings["paper_seeds"]
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError("paper cohort requires five distinct seeds")
        runs = [
            {"run_id": f"hint_{p:.2f}_seed{seed}", "seed": seed, "hint_probability": p, "has_hint": True}
            for p in RELIABILITIES for seed in seeds
        ] + [
            {"run_id": f"no_hint_seed{seed}", "seed": seed, "hint_probability": None, "has_hint": False}
            for seed in seeds
        ]
    else:
        seeds = settings["pilot_seeds"]
        if len(seeds) != 2 or len(set(seeds)) != 2:
            raise ValueError("pilot cohort requires two distinct seeds")
        runs = [
            {"run_id": f"hint_{p:.2f}_seed{seed}", "seed": seed, "hint_probability": p, "has_hint": True}
            for p, seed in zip((0.25, 0.90), seeds)
        ]
    if set(settings["paper_seeds"]) & set(settings["pilot_seeds"]):
        raise ValueError("pilot and paper training seeds must be disjoint")
    if settings["pilot_bank_seed"] == settings["paper_bank_seed"]:
        raise ValueError("pilot and paper banks must use different seeds")
    bank_seed = settings[f"{cohort}_bank_seed"]
    bank = make_bank(config, bank_seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bank_path = output / "bank.jsonl"
    with bank_path.open("x") as stream:
        for row in bank:
            write_row(stream, row)
    manifest = {
        "schema_version": 1, "cohort": cohort, "created_at": timestamp(),
        "paper_execution_enabled": settings.get("paper_execution_enabled", False) is True,
        "config": deepcopy(config), "runs": runs,
        "bank": artifact(output, bank_path, seed=bank_seed, groups=len(bank)),
        "observed_distributions": list(DISTRIBUTIONS),
        "price_distributions": list(PRICE_DISTRIBUTIONS), "preparation": provenance(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_study(study_dir, run_id):
    study_dir = Path(study_dir)
    manifest = read_json(study_dir / "manifest.json")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported study schema")
    validate_config(manifest["config"])
    matches = [r for r in manifest["runs"] if r["run_id"] == run_id]
    if len(matches) != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(f"unknown or invalid run ID: {run_id}")
    verify_artifact(study_dir, manifest["bank"])
    return manifest, matches[0]


def run_config(manifest, run):
    config = deepcopy(manifest["config"])
    config["RunConfig"].update(name=run["run_id"], seed=run["seed"])
    config["DataConfig"]["train_has_hint"] = run["has_hint"]
    if run["has_hint"]:
        config["DataConfig"]["train_hint_correct_probability"] = run["hint_probability"]
    return config


def bank_examples(study_dir, manifest, prompts):
    rows = list(read_rows(verify_artifact(study_dir, manifest["bank"])))
    if len(rows) != manifest["bank"]["groups"]:
        raise ValueError("bank group count mismatch")
    if type(prompts) is not int or not 0 < prompts <= len(rows) or prompts % 2:
        raise ValueError("measurement prompts must be positive, even, and within the saved bank capacity")
    return {name: [MCArithmeticExample(**row["variants"][name]) for row in rows[:prompts]] for name in DISTRIBUTIONS}
