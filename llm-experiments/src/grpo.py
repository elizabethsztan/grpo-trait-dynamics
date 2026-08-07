from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from .data import MCArithmeticExample, generate_examples
from .generation import GeneratedCompletion, generate_completions
from .logprobs import sequence_logprobs
from .metrics import summarize_trait_metrics
from .price import CumulativePriceTracker, price_covariance, price_stats
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
) -> tuple[dict, list[RolloutSample]]:
    import torch

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
    logprobs = sequence_logprobs(model, samples, tokenizer.pad_token_id, device=device, with_grad=True)
    pre_logprob_values = [float(value) for value in logprobs.detach().cpu().tolist()]
    advantage_tensor = torch.tensor(advantages, dtype=logprobs.dtype, device=logprobs.device)
    loss = -(advantage_tensor.detach() * logprobs).mean()

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
    optimizer.step()

    train_summary = summarize_trait_metrics(sample.traits for sample in samples)
    train_summary.update(
        {
            "loss": float(loss.detach().cpu()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
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
        cov = price_covariance(omega, trait_values)
        cumulative = tracker.update(distribution, trait_name, cov)
        stats = price_stats(omega, trait_values, cumulative=0.0)
        stats["cov_cum"] = cumulative
        block[trait_name] = stats

        if shuffled:
            shuffled_values = trait_values.copy()
            rng.shuffle(shuffled_values)
            shuffled_name = f"{trait_name}_shuffled"
            shuffled_cov = price_covariance(omega, shuffled_values)
            shuffled_cum = tracker.update(distribution, shuffled_name, shuffled_cov)
            block[shuffled_name] = {"cov_step": shuffled_cov, "cov_cum": shuffled_cum}
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
) -> dict:
    results = {}
    for name, dist_cfg in data_cfg["eval_distributions"].items():
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
