from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.activation_probe import build_activation_probe, make_completion_counterfactual_pairs, make_probe_pairs
from src.config import load_config, save_config
from src.data import generate_examples
from src.generation import configure_tokenizer_and_model, generate_completions
from src.grpo import RolloutSample
from src.metrics import summarize_trait_metrics
from src.traits import evaluate_completion_traits


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


def _auc(scores, labels) -> float | None:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
            total += 1
    return float(wins / total)


def _pairwise_accuracy(agree_scores, disagree_scores) -> float | None:
    agree = np.asarray(agree_scores, dtype=float)
    disagree = np.asarray(disagree_scores, dtype=float)
    if len(agree) == 0:
        return None
    return float(np.mean(agree > disagree))


def _correlation(values, labels) -> float | None:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(values) < 2 or float(np.std(values)) == 0.0 or float(np.std(labels)) == 0.0:
        return None
    return float(np.corrcoef(values, labels)[0, 1])


def _natural_rollout_score_metrics(scores, output_agreements, strict_valid) -> dict:
    rows = [
        (float(score), bool(agreement))
        for score, agreement, valid in zip(scores, output_agreements, strict_valid)
        if bool(valid)
    ]
    if not rows:
        return {"n_valid": 0, "correlation": None, "auc": None}
    valid_scores = [row[0] for row in rows]
    labels = [1 if row[1] else 0 for row in rows]
    return {
        "n_valid": len(rows),
        "correlation": _correlation(valid_scores, labels),
        "auc": _auc(valid_scores, labels),
    }


def _group_mean_diagnostics(rows: list[dict], group_key: str, score_key: str = "score") -> dict:
    groups = {}
    for row in rows:
        key = row.get(group_key)
        if key is None:
            continue
        groups.setdefault(str(key), []).append(float(row[score_key]))
    means = {key: float(np.mean(values)) for key, values in sorted(groups.items())}
    max_mean_gap = float(max(means.values()) - min(means.values())) if len(means) >= 2 else 0.0
    return {"means": means, "max_mean_gap": max_mean_gap}


def _line_plot(path: Path, x, ys, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in ys:
        ax.plot(x, values, marker="o", label=label)
    ax.set_xlabel("Hook layer")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hist_plot(path: Path, agree, disagree, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(agree, alpha=0.6, label="agree")
    ax.hist(disagree, alpha=0.6, label="disagree")
    ax.set_title(title)
    ax.set_xlabel("Activation agreement score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_plot(path: Path, labels, values, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_plot(path: Path, x, y, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _score_completion_counterfactual(model, tokenizer, probe, cfg, data_cfg, seed, device):
    pairs = make_completion_counterfactual_pairs(
        int(cfg.get("validation_pairs", 64)),
        seed=seed,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_phrases=data_cfg.get("eval_hint_phrases"),
    )
    agree_scores = probe.score_texts(model, tokenizer, [p.agree_prompt for p in pairs], [p.completion for p in pairs], device)
    disagree_scores = probe.score_texts(
        model,
        tokenizer,
        [p.disagree_prompt for p in pairs],
        [p.disagree_completion or p.completion for p in pairs],
        device,
    )
    agree = agree_scores.detach().float().cpu().numpy()
    disagree = disagree_scores.detach().float().cpu().numpy()
    labels = np.concatenate([np.ones_like(agree), np.zeros_like(disagree)])
    scores = np.concatenate([agree, disagree])
    return {
        "pairs": pairs,
        "agree": agree,
        "disagree": disagree,
        "metrics": {
            "mean_score_agree": float(np.mean(agree)),
            "mean_score_disagree": float(np.mean(disagree)),
            "score_gap": float(np.mean(agree) - np.mean(disagree)),
            "completion_counterfactual_auc": _auc(scores, labels),
            "completion_counterfactual_pairwise_accuracy": _pairwise_accuracy(agree, disagree),
        },
    }


def _score_prompt_counterfactual(model, tokenizer, probe, cfg, data_cfg, seed, device):
    pairs = make_probe_pairs(
        int(cfg.get("validation_pairs", 64)),
        seed=seed,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_phrases=data_cfg.get("eval_hint_phrases"),
    )
    agree_scores = probe.score_texts(model, tokenizer, [p.agree_prompt for p in pairs], [p.completion for p in pairs], device)
    disagree_scores = probe.score_texts(model, tokenizer, [p.disagree_prompt for p in pairs], [p.completion for p in pairs], device)
    agree = agree_scores.detach().float().cpu().numpy()
    disagree = disagree_scores.detach().float().cpu().numpy()
    labels = np.concatenate([np.ones_like(agree), np.zeros_like(disagree)])
    scores = np.concatenate([agree, disagree])
    return {
        "prompt_counterfactual_auc": _auc(scores, labels),
        "prompt_counterfactual_pairwise_accuracy": _pairwise_accuracy(agree, disagree),
        "prompt_counterfactual_score_gap": float(np.mean(agree) - np.mean(disagree)),
    }


def _score_letter_bias(completion_result) -> tuple[dict, list[dict]]:
    rows = []
    for pair, agree, disagree in zip(completion_result["pairs"], completion_result["agree"], completion_result["disagree"]):
        rows.append(
            {
                "completion_choice": pair.completion_choice,
                "user_hint": pair.completion_choice,
                "score": float(agree),
                "agreement": True,
            }
        )
        rows.append(
            {
                "completion_choice": pair.contrast_hint,
                "user_hint": pair.completion_choice,
                "score": float(disagree),
                "agreement": False,
            }
        )
    completion_choice = _group_mean_diagnostics(rows, "completion_choice")
    user_hint = _group_mean_diagnostics(rows, "user_hint")
    return {"completion_choice": completion_choice, "user_hint": user_hint}, rows


def _score_no_hint_control(model, tokenizer, probe, cfg, data_cfg, seed, device):
    from src.data import LETTERS

    n_examples = max(1, int(cfg.get("validation_pairs", 64)) // len(LETTERS))
    examples = generate_examples(
        n=n_examples,
        seed=seed,
        split="eval_no_hint",
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_correct_probability=0.0,
        has_hint=False,
        hint_phrases=data_cfg.get("eval_hint_phrases"),
    )
    prompts = []
    completions = []
    completion_choices = []
    for example in examples:
        for choice in LETTERS:
            prompts.append(example.prompt_text)
            completions.append(f"<answer>{choice}</answer>")
            completion_choices.append(choice)
    scores = probe.score_texts(model, tokenizer, prompts, completions, device).detach().float().cpu().numpy()
    rows = [{"completion_choice": choice, "score": float(score)} for choice, score in zip(completion_choices, scores)]
    by_choice = _group_mean_diagnostics(rows, "completion_choice")
    metrics = {
        "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
        "std_score": float(np.std(scores)) if len(scores) else 0.0,
        "max_abs_choice_mean": float(max((abs(value) for value in by_choice["means"].values()), default=0.0)),
        "completion_choice": by_choice,
    }
    return metrics, rows


def _score_natural_rollout(model, tokenizer, probe, config, data_cfg, seed, device):
    activation_cfg = config["ActivationProbeConfig"]
    rollout_examples = generate_examples(
        n=min(32, int(activation_cfg.get("validation_pairs", 64))),
        seed=seed,
        split="eval_wrong_hint",
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_correct_probability=0.0,
        has_hint=True,
        hint_phrases=data_cfg.get("eval_hint_phrases"),
    )
    samples = []
    for example in rollout_examples:
        completion = generate_completions(model, tokenizer, example.prompt_text, config["GenerationConfig"], 1, device)[0]
        traits = evaluate_completion_traits(
            completion.completion_text,
            example,
            completion.completion_token_length,
            stop_reason=completion.stop_reason,
            stopped_on_answer_tag=completion.stopped_on_answer_tag,
        )
        samples.append(RolloutSample(example=example, generated=completion, traits=traits))
    scores = probe.score_samples(model, samples, device).detach().float().cpu().numpy() if samples else np.asarray([], dtype=float)
    traits = [sample.traits for sample in samples]
    metrics = _natural_rollout_score_metrics(
        scores=scores,
        output_agreements=[trait.output_agreement for trait in traits],
        strict_valid=[trait.strict_valid for trait in traits],
    )
    metrics["summary"] = summarize_trait_metrics(traits)
    rows = [
        {
            "score": float(score),
            "output_agreement": bool(sample.traits.output_agreement),
            "strict_valid": bool(sample.traits.strict_valid),
            "completion": sample.generated.completion_text,
        }
        for score, sample in zip(scores, samples)
    ]
    return metrics, rows


def _selected_layer_result(layer_results: list[dict], selected_layer: int) -> dict:
    for item in layer_results:
        if int(item["layer"]) == int(selected_layer):
            return item
    available = [item.get("layer") for item in layer_results]
    raise ValueError(f"selected_hook_layer={selected_layer} was not evaluated; layer_sweep={available}")


def run_validation(config: dict) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_cfg = config["RunConfig"]
    model_cfg = config["ModelConfig"]
    activation_cfg = dict(config["ActivationProbeConfig"])
    data_cfg = config["DataConfig"]
    config["GenerationConfig"]["use_chat_template"] = bool(model_cfg.get("use_chat_template", True))
    out_dir = Path(run_cfg["results_dir"]) / f"{run_cfg['name']}_probe_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    device = _resolve_device(run_cfg)
    dtype = _resolve_dtype(run_cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], trust_remote_code=model_cfg.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    configure_tokenizer_and_model(tokenizer, model)

    layer_results = []
    examples_rows = []
    for layer in activation_cfg.get("layer_sweep", [activation_cfg.get("selected_hook_layer", 12)]):
        layer_cfg = dict(activation_cfg)
        layer_cfg["hook_layer"] = int(layer)
        probe = build_activation_probe(model, tokenizer, layer_cfg, data_cfg, int(run_cfg["seed"]), device)
        completion = _score_completion_counterfactual(
            model, tokenizer, probe, layer_cfg, data_cfg, int(run_cfg["seed"]) + 10000 + int(layer), device
        )
        prompt_metrics = _score_prompt_counterfactual(
            model, tokenizer, probe, layer_cfg, data_cfg, int(run_cfg["seed"]) + 20000 + int(layer), device
        )
        metrics = {"layer": int(layer), **completion["metrics"], **prompt_metrics}
        layer_results.append(metrics)
        if int(layer) == int(activation_cfg.get("selected_hook_layer", layer)):
            for pair, agree, disagree in zip(completion["pairs"], completion["agree"], completion["disagree"]):
                examples_rows.append(
                    {
                        "layer": int(layer),
                        "prompt": pair.agree_prompt,
                        "agree_completion": pair.completion,
                        "disagree_completion": pair.disagree_completion,
                        "agree_score": float(agree),
                        "disagree_score": float(disagree),
                    }
                )

    selected_layer = int(activation_cfg.get("selected_hook_layer", layer_results[0]["layer"]))
    selected = _selected_layer_result(layer_results, selected_layer)

    _line_plot(
        plots_dir / "auc_by_layer.png",
        [item["layer"] for item in layer_results],
        [("completion counterfactual AUC", [item["completion_counterfactual_auc"] for item in layer_results])],
        "AUC",
    )
    _line_plot(
        plots_dir / "pairwise_accuracy_by_layer.png",
        [item["layer"] for item in layer_results],
        [("completion counterfactual pairwise", [item["completion_counterfactual_pairwise_accuracy"] for item in layer_results])],
        "Pairwise accuracy",
    )
    selected_agree = [row["agree_score"] for row in examples_rows]
    selected_disagree = [row["disagree_score"] for row in examples_rows]
    _hist_plot(plots_dir / "score_separation_selected_layer.png", selected_agree, selected_disagree, "Selected layer score separation")
    selected_cfg = dict(activation_cfg)
    selected_cfg["hook_layer"] = selected_layer
    selected_probe = build_activation_probe(model, tokenizer, selected_cfg, data_cfg, int(run_cfg["seed"]), device)
    selected_completion = _score_completion_counterfactual(
        model, tokenizer, selected_probe, selected_cfg, data_cfg, int(run_cfg["seed"]) + 10000 + selected_layer, device
    )
    letter_bias, letter_bias_rows = _score_letter_bias(selected_completion)
    no_hint_control, no_hint_rows = _score_no_hint_control(
        model, tokenizer, selected_probe, selected_cfg, data_cfg, int(run_cfg["seed"]) + 30000, device
    )
    natural_rollout, natural_rollout_rows = _score_natural_rollout(
        model, tokenizer, selected_probe, config, data_cfg, int(run_cfg["seed"]) + 40000, device
    )
    completion_choice_means = letter_bias["completion_choice"]["means"]
    _bar_plot(
        plots_dir / "letter_bias_selected_layer.png",
        list(completion_choice_means.keys()),
        list(completion_choice_means.values()),
        "Mean activation agreement score",
        "Score by completion choice",
    )
    no_hint_means = no_hint_control["completion_choice"]["means"]
    _bar_plot(
        plots_dir / "no_hint_control_selected_layer.png",
        list(no_hint_means.keys()),
        list(no_hint_means.values()),
        "Mean activation agreement score",
        "No-hint control by completion choice",
    )
    _scatter_plot(
        plots_dir / "rollout_correlation_selected_layer.png",
        [1 if row["output_agreement"] else 0 for row in natural_rollout_rows if row["strict_valid"]],
        [row["score"] for row in natural_rollout_rows if row["strict_valid"]],
        "Output agreement",
        "Activation agreement score",
        "Natural rollout score correlation",
    )

    payload = {
        "run_name": run_cfg["name"],
        "layer_results": layer_results,
        "selected_layer": selected,
        "selected_layer_diagnostics": {
            "letter_bias": letter_bias,
            "no_hint_control": no_hint_control,
            "natural_rollout": natural_rollout,
        },
    }
    (out_dir / "activation_probe_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / "examples.jsonl").open("w") as f:
        for row in examples_rows:
            f.write(json.dumps(row) + "\n")
        for row in letter_bias_rows:
            f.write(json.dumps({"diagnostic": "letter_bias", **row}) + "\n")
        for row in no_hint_rows:
            f.write(json.dumps({"diagnostic": "no_hint_control", **row}) + "\n")
        for row in natural_rollout_rows:
            f.write(json.dumps({"diagnostic": "natural_rollout", **row}) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir = run_validation(config)
    print(f"wrote activation probe validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
