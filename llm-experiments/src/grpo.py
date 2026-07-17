from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

import numpy as np

from .data import MCArithmeticExample, generate_examples
from .generation import GeneratedCompletion, generate_completions
from .logprobs import sequence_logprobs, token_logprob_batch
from .metrics import summarize_trait_metrics
from .price import CumulativePriceTracker, price_stats
from .traits import CompletionTraitMetrics, evaluate_completion_traits


@dataclass
class RolloutSample:
    example: MCArithmeticExample
    generated: GeneratedCompletion
    traits: CompletionTraitMetrics
    pre_logprob: float | None = None
    post_logprob: float | None = None

    @property
    def prompt_ids(self):
        return self.generated.prompt_ids

    @property
    def completion_ids(self):
        return self.generated.completion_ids


def compute_group_advantages(rewards: np.ndarray, eps: float) -> np.ndarray:
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    advantages = (rewards - mean) / (std + eps)
    return np.where(std < eps, 0.0, advantages)


def clipped_grpo_loss(
    current_token_logprobs,
    old_token_logprobs,
    mask,
    advantages,
    clip_range: float,
    normalize_loss_by_tokens: bool = True,
    reference_token_logprobs=None,
    kl_coef: float = 0.0,
):
    import torch

    mask_f = mask.to(dtype=current_token_logprobs.dtype, device=current_token_logprobs.device)
    old_token_logprobs = old_token_logprobs.to(current_token_logprobs.device)
    advantages = advantages.to(dtype=current_token_logprobs.dtype, device=current_token_logprobs.device)
    while advantages.ndim < current_token_logprobs.ndim:
        advantages = advantages.unsqueeze(-1)

    ratios = torch.exp(current_token_logprobs - old_token_logprobs)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)
    objective = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    masked_objective = objective * mask_f
    if normalize_loss_by_tokens:
        policy_loss = -masked_objective.sum() / mask_f.sum().clamp_min(1.0)
    else:
        per_sequence = masked_objective.sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        policy_loss = -per_sequence.mean()

    reference_kl = torch.zeros((), dtype=current_token_logprobs.dtype, device=current_token_logprobs.device)
    if reference_token_logprobs is not None and kl_coef > 0:
        reference_token_logprobs = reference_token_logprobs.to(current_token_logprobs.device)
        reference_log_ratio = reference_token_logprobs - current_token_logprobs
        per_token_kl = torch.exp(reference_log_ratio) - reference_log_ratio - 1.0
        reference_kl = (per_token_kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    loss = policy_loss + float(kl_coef) * reference_kl

    with torch.no_grad():
        active_ratios = ratios[mask.to(device=ratios.device)]
        if active_ratios.numel():
            clip_fraction = ((active_ratios < 1.0 - clip_range) | (active_ratios > 1.0 + clip_range)).float().mean()
            stats = {
                "ratio_mean": float(active_ratios.mean().detach().cpu()),
                "ratio_min": float(active_ratios.min().detach().cpu()),
                "ratio_max": float(active_ratios.max().detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
                "approx_kl": float(((old_token_logprobs - current_token_logprobs) * mask_f).sum().detach().cpu() / mask_f.sum().clamp_min(1.0)),
                "reference_kl": float(reference_kl.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
        else:
            stats = {
                "ratio_mean": 0.0,
                "ratio_min": 0.0,
                "ratio_max": 0.0,
                "clip_fraction": 0.0,
                "approx_kl": 0.0,
                "reference_kl": 0.0,
                "policy_loss": 0.0,
                "total_loss": float(loss.detach().cpu()),
            }
    return loss, stats


def pad_token_logprobs_to_width(token_logprobs, width: int):
    import torch

    current_width = token_logprobs.shape[1]
    if current_width == width:
        return token_logprobs
    if current_width > width:
        return token_logprobs[:, :width]
    pad = torch.zeros(
        (token_logprobs.shape[0], width - current_width),
        dtype=token_logprobs.dtype,
        device=token_logprobs.device,
    )
    return torch.cat([token_logprobs, pad], dim=1)


def _disable_adapters_if_available(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    return nullcontext()


def make_examples_for_distribution(
    cfg: dict,
    data_cfg: dict,
    n: int,
    seed: int,
    split: str,
) -> list[MCArithmeticExample]:
    return generate_examples(
        n=n,
        seed=seed,
        split=split,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_correct_probability=float(cfg.get("hint_correct_probability", data_cfg.get("train_hint_correct_probability", 0.9))),
        has_hint=bool(cfg.get("has_hint", True)),
        hint_phrases=data_cfg.get("eval_hint_phrases" if split.startswith("eval") else "train_hint_phrases"),
    )


def score_activation_for_samples(model, tokenizer, activation_probe, samples: list[RolloutSample], device) -> list[float | None]:
    if activation_probe is None or not samples:
        return [None for _ in samples]
    if hasattr(activation_probe, "score_samples"):
        scores = activation_probe.score_samples(model, samples, device)
    else:
        scores = activation_probe.score_texts(
            model,
            tokenizer,
            [sample.example.prompt_text for sample in samples],
            [sample.generated.completion_text for sample in samples],
            device,
        )
    return [float(score) for score in scores.detach().cpu()]


def sample_rollouts(
    model,
    tokenizer,
    examples: list[MCArithmeticExample],
    generation_cfg: dict,
    completions_per_prompt: int,
    device,
    activation_probe=None,
) -> list[RolloutSample]:
    samples: list[RolloutSample] = []
    for example in examples:
        generated = generate_completions(
            model,
            tokenizer,
            example.prompt_text,
            generation_cfg,
            num_return_sequences=completions_per_prompt,
            device=device,
        )
        for completion in generated:
            samples.append(
                RolloutSample(
                    example=example,
                    generated=completion,
                    traits=evaluate_completion_traits(
                        completion.completion_text,
                        example,
                        completion_token_length=completion.completion_token_length,
                        stop_reason=completion.stop_reason,
                        stopped_on_answer_tag=completion.stopped_on_answer_tag,
                    ),
                )
            )

    activation_scores = score_activation_for_samples(model, tokenizer, activation_probe, samples, device)
    if activation_probe is not None:
        samples = [
            RolloutSample(
                example=sample.example,
                generated=sample.generated,
                traits=evaluate_completion_traits(
                    sample.generated.completion_text,
                    sample.example,
                    completion_token_length=sample.generated.completion_token_length,
                    stop_reason=sample.generated.stop_reason,
                    stopped_on_answer_tag=sample.generated.stopped_on_answer_tag,
                    activation_agreement=score,
                ),
                pre_logprob=sample.pre_logprob,
                post_logprob=sample.post_logprob,
            )
            for sample, score in zip(samples, activation_scores)
        ]
    return samples


def train_grpo_step(
    model,
    tokenizer,
    optimizer,
    examples: list[MCArithmeticExample],
    generation_cfg: dict,
    group_size: int,
    eps: float,
    max_grad_norm: float,
    device,
    activation_probe=None,
    grpo_cfg: dict | None = None,
) -> tuple[dict, list[RolloutSample]]:
    import torch

    grpo_cfg = grpo_cfg or {}
    samples = sample_rollouts(
        model,
        tokenizer,
        examples,
        generation_cfg,
        completions_per_prompt=group_size,
        device=device,
        activation_probe=activation_probe,
    )
    rewards = np.asarray([sample.traits.reward for sample in samples], dtype=float).reshape(len(examples), group_size)
    advantages = compute_group_advantages(rewards, eps=eps).reshape(-1)
    old_batch = token_logprob_batch(model, samples, tokenizer.pad_token_id, device=device, with_grad=False)
    pre_logprob_values = [float(value) for value in old_batch.sequence_logprobs.detach().cpu().tolist()]
    advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    clip_range = float(grpo_cfg.get("clip_range", 0.2))
    num_policy_epochs = int(grpo_cfg.get("num_policy_epochs", 1))
    normalize_loss_by_tokens = bool(grpo_cfg.get("normalize_loss_by_tokens", True))
    use_reference_kl = bool(grpo_cfg.get("use_reference_kl", False))
    kl_coef = float(grpo_cfg.get("kl_coef", 0.0))
    minibatch_size = grpo_cfg.get("minibatch_size")
    minibatch_size = len(samples) if minibatch_size is None else int(minibatch_size)
    last_loss = torch.zeros((), dtype=torch.float32, device=device)
    last_stats = {
        "ratio_mean": 1.0,
        "ratio_min": 1.0,
        "ratio_max": 1.0,
        "clip_fraction": 0.0,
        "approx_kl": 0.0,
        "reference_kl": 0.0,
        "policy_loss": 0.0,
        "total_loss": 0.0,
    }
    reference_batch = None
    if use_reference_kl and kl_coef > 0:
        with _disable_adapters_if_available(model):
            reference_batch = token_logprob_batch(model, samples, tokenizer.pad_token_id, device=device, with_grad=False)
        reference_token_logprobs = pad_token_logprobs_to_width(
            reference_batch.token_logprobs,
            width=old_batch.token_logprobs.shape[1],
        )
    else:
        reference_token_logprobs = None
    for _epoch in range(num_policy_epochs):
        for start in range(0, len(samples), minibatch_size):
            end = min(start + minibatch_size, len(samples))
            current = token_logprob_batch(model, samples[start:end], tokenizer.pad_token_id, device=device, with_grad=True)
            current_token_logprobs = pad_token_logprobs_to_width(
                current.token_logprobs,
                width=old_batch.token_logprobs.shape[1],
            )
            minibatch_reference_token_logprobs = None
            if reference_token_logprobs is not None:
                minibatch_reference_token_logprobs = reference_token_logprobs[start:end]
            loss, stats = clipped_grpo_loss(
                current_token_logprobs,
                old_batch.token_logprobs[start:end],
                old_batch.mask[start:end],
                advantage_tensor[start:end],
                clip_range=clip_range,
                normalize_loss_by_tokens=normalize_loss_by_tokens,
                reference_token_logprobs=minibatch_reference_token_logprobs,
                kl_coef=kl_coef,
            )
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
            optimizer.step()
            last_loss = loss.detach()
            last_stats = stats

    train_summary = summarize_trait_metrics(sample.traits for sample in samples)
    train_summary.update(
        {
            "loss": float(last_loss.cpu()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "advantage_mean": float(np.mean(advantages)),
            "advantage_std": float(np.std(advantages)),
            "ratio_mean": last_stats["ratio_mean"],
            "ratio_min": last_stats["ratio_min"],
            "ratio_max": last_stats["ratio_max"],
            "clip_fraction": last_stats["clip_fraction"],
            "approx_kl": last_stats["approx_kl"],
            "reference_kl": last_stats["reference_kl"],
            "policy_loss": last_stats["policy_loss"],
            "total_loss": last_stats["total_loss"],
        }
    )
    samples = [
        RolloutSample(
            example=sample.example,
            generated=sample.generated,
            traits=sample.traits,
            pre_logprob=pre_logprob,
            post_logprob=sample.post_logprob,
        )
        for sample, pre_logprob in zip(samples, pre_logprob_values)
    ]
    return train_summary, samples


def attach_logprobs(model, tokenizer, samples: list[RolloutSample], device, field: str) -> list[RolloutSample]:
    scores = sequence_logprobs(model, samples, tokenizer.pad_token_id, device=device, with_grad=False)
    updated = []
    for sample, score in zip(samples, scores.detach().cpu().tolist()):
        kwargs = {
            "example": sample.example,
            "generated": sample.generated,
            "traits": sample.traits,
            "pre_logprob": sample.pre_logprob,
            "post_logprob": sample.post_logprob,
        }
        kwargs[field] = float(score)
        updated.append(RolloutSample(**kwargs))
    return updated


def compute_price_block(samples: list[RolloutSample], tracker: CumulativePriceTracker, distribution: str, rng: random.Random, shuffled: bool):
    if not samples:
        return {}
    pre = np.asarray([sample.pre_logprob for sample in samples], dtype=float)
    post = np.asarray([sample.post_logprob for sample in samples], dtype=float)
    omega = np.exp(post - pre)
    traits = {
        "output_agreement": np.asarray([sample.traits.output_agreement for sample in samples], dtype=float),
    }
    if any(sample.traits.activation_agreement is not None for sample in samples):
        traits["activation_agreement"] = np.asarray(
            [sample.traits.activation_agreement or 0.0 for sample in samples],
            dtype=float,
        )

    block = {}
    for trait_name, trait_values in traits.items():
        stats = price_stats(omega, trait_values, cumulative=0.0)
        cumulative = tracker.update_price(distribution, trait_name, stats["raw_cov_step"], stats["sn_step"])
        stats["raw_cov_cum"] = cumulative["raw_cov_cum"]
        stats["sn_cum"] = cumulative["sn_cum"]
        stats["cov_step"] = stats["raw_cov_step"]
        stats["cov_cum"] = stats["raw_cov_cum"]
        block[trait_name] = stats

        if shuffled:
            shuffled_values = trait_values.copy()
            rng.shuffle(shuffled_values)
            shuffled_name = f"{trait_name}_shuffled"
            shuffled_stats = price_stats(omega, shuffled_values, cumulative=0.0)
            shuffled_cum = tracker.update_price(distribution, shuffled_name, shuffled_stats["raw_cov_step"], shuffled_stats["sn_step"])
            shuffled_stats["raw_cov_cum"] = shuffled_cum["raw_cov_cum"]
            shuffled_stats["sn_cum"] = shuffled_cum["sn_cum"]
            shuffled_stats["cov_step"] = shuffled_stats["raw_cov_step"]
            shuffled_stats["cov_cum"] = shuffled_stats["raw_cov_cum"]
            block[shuffled_name] = shuffled_stats
    return block


def observed_eval(
    model,
    tokenizer,
    data_cfg: dict,
    observed_cfg: dict,
    generation_cfg: dict,
    device,
    step: int,
    activation_probe=None,
    examples_by_distribution: dict | None = None,
) -> dict:
    results = {}
    for name, dist_cfg in data_cfg["eval_distributions"].items():
        if examples_by_distribution and name in examples_by_distribution:
            examples = examples_by_distribution[name]
        else:
            name_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(name)) % 997
            examples = make_examples_for_distribution(
                dist_cfg,
                data_cfg,
                n=int(observed_cfg.get("prompts_per_distribution", 4)),
                seed=int(data_cfg.get("seed_offset", 0)) + 100000 + step * 1000 + name_offset,
                split=name,
            )
        samples = sample_rollouts(
            model,
            tokenizer,
            examples,
            generation_cfg,
            completions_per_prompt=int(observed_cfg.get("completions_per_prompt", 1)),
            device=device,
            activation_probe=activation_probe,
        )
        results[name] = summarize_trait_metrics(sample.traits for sample in samples)
    return results
