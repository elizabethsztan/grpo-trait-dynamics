from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, save_config
from src.data import generate_examples
from src.generation import configure_tokenizer_and_model, generate_completions
from src.metrics import summarize_trait_metrics
from src.traits import evaluate_completion_traits


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _validate_examples(examples) -> None:
    for example in examples:
        if example.gold_choice not in example.options:
            raise AssertionError(f"gold_choice missing from options: {example.problem_id}")
        if example.options[example.gold_choice] != example.gold_value:
            raise AssertionError(f"gold value not at gold choice: {example.problem_id}")
        if len(set(example.options.values())) != 4:
            raise AssertionError(f"duplicate option values: {example.problem_id}")
        if example.user_hint is None and example.hint_is_correct is not None:
            raise AssertionError(f"no-hint example has hint_is_correct: {example.problem_id}")


def build_calibration_grid(config: dict) -> list[tuple[str, dict]]:
    difficulties = ["easy", "medium", "hard"]
    reliabilities = [0.0, 0.5, 0.9]
    grid = []
    for difficulty in difficulties:
        for hint_correct_probability in reliabilities:
            for phrase_set in ("train", "eval"):
                grid.append(
                    (
                        f"{difficulty}_{phrase_set}_phrases_hint_{hint_correct_probability:.1f}",
                        {
                            "difficulty": difficulty,
                            "hint_correct_probability": hint_correct_probability,
                            "has_hint": True,
                            "phrase_set": phrase_set,
                        },
                    )
                )
    grid.append(("medium_no_hint", {"difficulty": "medium", "hint_correct_probability": 0.0, "has_hint": False}))
    return grid


def run_dry_calibration(config: dict, run_dir: Path) -> dict:
    data_cfg = config["DataConfig"]
    observed_cfg = config["ObservedEvalConfig"]
    rows = []
    metrics = {}
    for name, grid_cfg in build_calibration_grid(config):
        examples = generate_examples(
            n=int(observed_cfg.get("prompts_per_distribution", 4)),
            seed=int(config["RunConfig"]["seed"]) + len(rows),
            split=f"calibration_{name}",
            difficulty=grid_cfg["difficulty"],
            hint_correct_probability=grid_cfg["hint_correct_probability"],
            has_hint=grid_cfg["has_hint"],
            hint_phrases=data_cfg.get("train_hint_phrases" if grid_cfg.get("phrase_set") == "train" else "eval_hint_phrases"),
        )
        _validate_examples(examples)
        rows.extend(example.to_json_dict() for example in examples)
        metrics[name] = {
            "num_examples": len(examples),
            "dry_run": True,
            "accuracy": None,
            "agreement_rate": None,
            "invalid_output_rate": None,
            "no_hint_accuracy": None if grid_cfg["has_hint"] else None,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    (run_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_jsonl(run_dir / "calibration_examples.jsonl", rows)
    return metrics


def run_model_calibration(config: dict, run_dir: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_cfg = config["RunConfig"]
    model_cfg = config["ModelConfig"]
    data_cfg = config["DataConfig"]
    observed_cfg = config["ObservedEvalConfig"]
    device = "cuda" if run_cfg.get("device") == "auto" and torch.cuda.is_available() else (
        "cpu" if run_cfg.get("device") == "auto" else run_cfg["device"]
    )
    dtype = torch.bfloat16 if run_cfg.get("dtype") == "bf16" else None
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], trust_remote_code=model_cfg.get("trust_remote_code", False))
    model_kwargs = {"trust_remote_code": model_cfg.get("trust_remote_code", False)}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name"], **model_kwargs).to(device)
    model.eval()
    configure_tokenizer_and_model(tokenizer, model)

    rows = []
    metrics = {}
    for idx, (name, grid_cfg) in enumerate(build_calibration_grid(config)):
        examples = generate_examples(
            n=int(observed_cfg.get("prompts_per_distribution", 4)),
            seed=int(run_cfg["seed"]) + idx,
            split=f"calibration_{name}",
            difficulty=grid_cfg["difficulty"],
            hint_correct_probability=grid_cfg["hint_correct_probability"],
            has_hint=grid_cfg["has_hint"],
            hint_phrases=data_cfg.get("train_hint_phrases" if grid_cfg.get("phrase_set") == "train" else "eval_hint_phrases"),
        )
        _validate_examples(examples)
        traits = []
        for example in examples:
            completion = generate_completions(model, tokenizer, example.prompt_text, config["GenerationConfig"], 1, device)[0]
            metric = evaluate_completion_traits(
                completion.completion_text,
                example,
                completion_token_length=completion.completion_token_length,
            )
            traits.append(metric)
            row = example.to_json_dict()
            row.update({"completion_text": completion.completion_text, "metrics": metric.to_json_dict()})
            rows.append(row)
        summary = summarize_trait_metrics(traits)
        summary["no_hint_accuracy"] = summary["accuracy"] if not grid_cfg["has_hint"] else None
        metrics[name] = summary

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    (run_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_jsonl(run_dir / "calibration_examples.jsonl", rows)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    name = f"{config['RunConfig']['name']}_calibration"
    run_dir = Path(config["RunConfig"]["results_dir"]) / name
    if args.dry_run:
        run_dry_calibration(config, run_dir)
    else:
        run_model_calibration(config, run_dir)
    print(f"wrote calibration outputs to {run_dir}")


if __name__ == "__main__":
    main()
