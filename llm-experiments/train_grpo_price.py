from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from src.activation_probe import build_activation_probe, make_probe_pairs
from src.config import load_config, save_config
from src.data import generate_examples
from src.generation import configure_tokenizer_and_model
from src.grpo import (
    attach_logprobs,
    compute_price_block,
    make_examples_for_distribution,
    observed_eval,
    sample_rollouts,
    train_grpo_step,
)
from src.lora_freeze import apply_lora_above_hook
from src.price import CumulativePriceTracker


def _resolve_device(run_cfg):
    import torch

    requested = run_cfg.get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _resolve_dtype(run_cfg):
    import torch

    dtype = run_cfg.get("dtype", "auto")
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _write_jsonl_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _activation_scores_to_numpy(scores):
    return scores.detach().float().cpu().numpy()


def _record_step_for_update(update_index: int) -> int:
    return update_index + 1


def _latest_eval_wrong_hint(metrics: list[dict]) -> dict:
    for record in reversed(metrics):
        observed_wrong = record.get("observed_eval", {}).get("eval_wrong_hint", {})
        if observed_wrong:
            return observed_wrong
    return {}


def _write_rollout_examples(run_dir: Path, step: int, samples, logging_cfg: dict) -> None:
    if not logging_cfg.get("enabled", True):
        return
    every = int(logging_cfg.get("every", 1))
    if every <= 0 or step % every != 0:
        return
    limit = int(logging_cfg.get("max_examples_per_step", 16))
    path = run_dir / "examples" / f"train_rollouts_step_{step:04d}.jsonl"
    rows = []
    for sample in samples[:limit]:
        rows.append(
            {
                "problem_id": sample.example.problem_id,
                "prompt_text": sample.example.prompt_text,
                "gold_choice": sample.example.gold_choice,
                "user_hint": sample.example.user_hint,
                "hint_is_correct": sample.example.hint_is_correct,
                "completion_text": sample.generated.completion_text,
                "completion_ids": sample.generated.completion_ids,
                "metrics": sample.traits.to_json_dict(),
            }
        )
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _generate_train_examples(config: dict, step: int):
    run_cfg = config["RunConfig"]
    data_cfg = config["DataConfig"]
    train_cfg = config["TrainConfig"]
    return generate_examples(
        n=int(train_cfg["train_prompts_per_step"]),
        seed=int(run_cfg["seed"]) + step * 1009,
        split="train_high_hint",
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_correct_probability=float(data_cfg.get("train_hint_correct_probability", 0.9)),
        has_hint=True,
        hint_phrases=data_cfg.get("train_hint_phrases"),
    )


def _make_price_rollouts(model, tokenizer, config, step: int, device, activation_probe):
    run_cfg = config["RunConfig"]
    data_cfg = config["DataConfig"]
    price_cfg = config["PriceConfig"]
    rollouts = {}
    for idx, dist_name in enumerate(price_cfg.get("eval_distributions", [])):
        dist_cfg = data_cfg["eval_distributions"][dist_name]
        examples = make_examples_for_distribution(
            dist_cfg,
            data_cfg,
            n=int(price_cfg.get("prompts_per_distribution", 2)),
            seed=int(run_cfg["seed"]) + 500000 + step * 1000 + idx,
            split=dist_name,
        )
        samples = sample_rollouts(
            model,
            tokenizer,
            examples,
            config["GenerationConfig"],
            completions_per_prompt=int(price_cfg.get("completions_per_prompt", 1)),
            device=device,
            activation_probe=activation_probe,
        )
        rollouts[dist_name] = attach_logprobs(model, tokenizer, samples, device, "pre_logprob")
    return rollouts


def _activation_invariance(model, tokenizer, probe, cfg, data_cfg, seed: int, device, baseline=None):
    if probe is None:
        return None, None
    pairs = make_probe_pairs(
        int(cfg.get("invariance_bank_size", 16)),
        seed=seed + 700000,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_phrases=data_cfg.get("train_hint_phrases"),
    )
    prompts = [pair.agree_prompt for pair in pairs]
    completions = [pair.completion for pair in pairs]
    scores = _activation_scores_to_numpy(probe.score_texts(model, tokenizer, prompts, completions, device))
    if baseline is None:
        return scores, {"max_abs": 0.0, "mean_abs": 0.0}
    delta = np.abs(scores - baseline)
    return baseline, {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def run_training(config: dict) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_cfg = config["RunConfig"]
    model_cfg = config["ModelConfig"]
    train_cfg = config["TrainConfig"]
    price_cfg = config["PriceConfig"]
    data_cfg = config["DataConfig"]
    activation_cfg = dict(config["ActivationProbeConfig"])
    activation_cfg["hook_layer"] = config["LoRAConfig"]["hook_layer"]

    run_dir = Path(run_cfg["results_dir"]) / run_cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "examples").mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    save_config(config, run_dir / "config.yaml")

    torch.manual_seed(int(run_cfg["seed"]))
    random.seed(int(run_cfg["seed"]))
    np.random.seed(int(run_cfg["seed"]))
    device = _resolve_device(run_cfg)
    dtype = _resolve_dtype(run_cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], trust_remote_code=model_cfg.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        torch_dtype=dtype,
    ).to(device)
    configure_tokenizer_and_model(tokenizer, model)
    if model_cfg.get("use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model = apply_lora_above_hook(model, config["LoRAConfig"]).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    activation_probe = None
    invariance_baseline = None
    if activation_cfg.get("enabled", True):
        activation_probe = build_activation_probe(model, tokenizer, activation_cfg, data_cfg, int(run_cfg["seed"]), device)
        invariance_baseline, _ = _activation_invariance(
            model, tokenizer, activation_probe, activation_cfg, data_cfg, int(run_cfg["seed"]), device
        )

    price_tracker = CumulativePriceTracker()
    rng = random.Random(int(run_cfg["seed"]) + 4242)
    observed_baselines = {}

    def eval_observed_for_step(step: int) -> dict:
        observed = {}
        if step == 0 or step % int(config["ObservedEvalConfig"].get("eval_every", 1)) == 0:
            observed = observed_eval(
                model,
                tokenizer,
                data_cfg,
                config["ObservedEvalConfig"],
                config["GenerationConfig"],
                device,
                step,
                activation_probe=activation_probe,
            )
            for dist_name, values in observed.items():
                observed_baselines.setdefault(
                    dist_name,
                    {
                        "agreement_rate": values.get("agreement_rate", 0.0),
                        "mean_activation_agreement_score": values.get("mean_activation_agreement_score", 0.0),
                    },
                )
                values["output_agreement_observed_drift"] = (
                    values.get("agreement_rate", 0.0) - observed_baselines[dist_name]["agreement_rate"]
                )
                values["activation_agreement_observed_drift"] = (
                    values.get("mean_activation_agreement_score", 0.0)
                    - observed_baselines[dist_name]["mean_activation_agreement_score"]
                )
        return observed

    def eval_invariance_for_step(step: int, baseline):
        activation_invariance = None
        if activation_probe is not None and step % int(activation_cfg.get("invariance_every", 1)) == 0:
            baseline, activation_invariance = _activation_invariance(
                model,
                tokenizer,
                activation_probe,
                activation_cfg,
                data_cfg,
                int(run_cfg["seed"]),
                device,
                baseline=baseline,
            )
            if run_cfg.get("debug", False) and activation_invariance["max_abs"] > float(activation_cfg["invariance_assert_threshold"]):
                raise AssertionError(f"activation invariance failed: {activation_invariance}")
        return baseline, activation_invariance

    observed = eval_observed_for_step(0)
    invariance_baseline, activation_invariance = eval_invariance_for_step(0, invariance_baseline)
    _write_jsonl_line(
        metrics_path,
        {
            "step": 0,
            "train": {},
            "price": {},
            "observed_eval": observed,
            "activation_invariance": activation_invariance or {},
        },
    )

    for update_idx in range(int(train_cfg["num_steps"])):
        price_rollouts = {}
        if price_cfg.get("enabled", True):
            price_rollouts = _make_price_rollouts(model, tokenizer, config, update_idx, device, activation_probe)

        train_examples = _generate_train_examples(config, update_idx)
        train_summary, train_samples = train_grpo_step(
            model,
            tokenizer,
            optimizer,
            train_examples,
            config["GenerationConfig"],
            group_size=int(train_cfg["group_size"]),
            eps=float(train_cfg.get("eps", 1e-8)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            device=device,
            activation_probe=activation_probe,
        )
        _write_rollout_examples(run_dir, update_idx, train_samples, config["ExampleLoggingConfig"])

        price_block = {}
        for dist_name, samples in price_rollouts.items():
            post_samples = attach_logprobs(model, tokenizer, samples, device, "post_logprob")
            price_block[dist_name] = compute_price_block(
                post_samples,
                price_tracker,
                dist_name,
                rng,
                shuffled=bool(price_cfg.get("compute_shuffled_null", True)),
            )

        if price_cfg.get("compute_reused_rollout_diagnostic", True):
            reused = attach_logprobs(model, tokenizer, train_samples, device, "post_logprob")
            diag = compute_price_block(reused, CumulativePriceTracker(), "train_high_hint_reused", rng, shuffled=False)
            train_summary["reused_train_rollout_cov_output_agreement"] = diag.get("output_agreement", {}).get("cov_step", 0.0)
            train_summary["reused_train_rollout_cov_activation_agreement"] = diag.get("activation_agreement", {}).get("cov_step", 0.0)

        step = _record_step_for_update(update_idx)
        observed = eval_observed_for_step(step)
        invariance_baseline, activation_invariance = eval_invariance_for_step(step, invariance_baseline)
        record = {
            "step": step,
            "train": train_summary,
            "price": price_block,
            "observed_eval": observed,
            "activation_invariance": activation_invariance or {},
        }
        _write_jsonl_line(metrics_path, record)

    adapter_dir = run_dir / "adapter" / "final"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    summary = {"run_name": run_cfg["name"], "num_steps": train_cfg["num_steps"], "metrics_path": str(metrics_path)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


def run_reliability_sweep(config: dict) -> Path:
    sweep_cfg = config["ReliabilitySweepConfig"]
    sweep_name = config["RunConfig"]["name"]
    sweep_dir = Path(config["RunConfig"]["results_dir"]) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for probability in sweep_cfg["train_hint_correct_probabilities"]:
        run_config = json.loads(json.dumps(config))
        run_config["RunConfig"]["results_dir"] = str(sweep_dir)
        run_config["RunConfig"]["name"] = f"hint_reliability_{str(probability).replace('.', '_')}"
        run_config["DataConfig"]["train_hint_correct_probability"] = float(probability)
        run_config["DataConfig"]["eval_distributions"]["train_high_hint"]["hint_correct_probability"] = float(probability)
        run_dir = run_training(run_config)
        metrics_path = run_dir / "metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        observed_wrong = _latest_eval_wrong_hint(metrics)
        price_wrong = {}
        for record in reversed(metrics):
            price_wrong = record.get("price", {}).get("eval_wrong_hint", {})
            if price_wrong:
                break
        runs.append(
            {
                "train_hint_correct_probability": float(probability),
                "run_dir": str(run_dir),
                "final_wrong_hint_agreement_rate": observed_wrong.get("wrong_hint_agreement_rate"),
                "final_sycophantic_error_rate": observed_wrong.get("sycophantic_error_rate"),
                "final_output_agreement_price_cum": price_wrong.get("output_agreement", {}).get("cov_cum"),
            }
        )
    summary = {
        "sweep_name": sweep_name,
        "runs": runs,
        "expected_pattern": sweep_cfg.get("expected_pattern", {}),
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return sweep_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if "ReliabilitySweepConfig" in config:
        run_dir = run_reliability_sweep(config)
        print(f"wrote reliability sweep outputs to {run_dir}")
    else:
        run_dir = run_training(config)
        print(f"wrote training outputs to {run_dir}")


if __name__ == "__main__":
    main()
