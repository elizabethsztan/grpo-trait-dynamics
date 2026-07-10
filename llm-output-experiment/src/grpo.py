from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from .data import MCArithmeticExample, generate_examples
from .generation import GeneratedCompletion, generate_completions
from .logprobs import sequence_logprobs, token_logprobs
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


def sample_rollouts(
    model,
    tokenizer,
    examples: list[MCArithmeticExample],
    generation_cfg: dict,
    completions_per_prompt: int,
    device,
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
    return samples


def _optimizer_step(model, optimizer, loss, max_grad_norm) -> None:
    import torch

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
    optimizer.step()


def _vanilla_update(
    model,
    tokenizer,
    optimizer,
    samples: list[RolloutSample],
    advantages: np.ndarray,
    max_grad_norm: float,
    device,
    kl_coef: float,
) -> tuple[dict, list[float]]:
    """One REINFORCE-style step on the whole batch: no ratio, so nothing restrains step 1."""
    import torch

    token_lp, mask = token_logprobs(model, samples, tokenizer.pad_token_id, device=device, with_grad=True)
    logprobs = (token_lp * mask).sum(dim=1)
    pre_logprob_values = [float(value) for value in logprobs.detach().cpu().tolist()]
    advantage_tensor = torch.tensor(advantages, dtype=logprobs.dtype, device=logprobs.device)
    policy_loss = -(advantage_tensor.detach() * logprobs).mean()

    # KL(pi_theta || pi_ref) to the frozen base model (LoRA off), k3 estimator, token-averaged.
    # Restrains how far each update moves the policy -- the Price estimator needs omega ~ 1.
    kl_value = 0.0
    loss = policy_loss
    if kl_coef > 0:
        with torch.no_grad(), model.disable_adapter():
            ref_lp, _ = token_logprobs(model, samples, tokenizer.pad_token_id, device=device, with_grad=False)
        log_ratio = (ref_lp - token_lp).float()  # fp32: exp() overflows in bf16
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
        kl = (per_token_kl * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        loss = policy_loss + float(kl_coef) * kl.to(policy_loss.dtype)
        kl_value = float(kl.detach().cpu())

    _optimizer_step(model, optimizer, loss, max_grad_norm)
    summary = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": kl_value,
    }
    return summary, pre_logprob_values


def _clipped_update(
    model,
    tokenizer,
    optimizer,
    samples: list[RolloutSample],
    advantages: np.ndarray,
    max_grad_norm: float,
    device,
    kl_coef: float,
    clip_range: float,
    minibatch_size: int | None,
    num_policy_epochs: int,
) -> tuple[dict, list[float]]:
    """PPO-style clipped update over contiguous minibatches.

    The first minibatch of the first epoch has ratio == 1 everywhere, so clipping cannot
    restrain it -- only the later minibatches see a moved policy.
    """
    import torch

    old_token_lp, old_mask = token_logprobs(model, samples, tokenizer.pad_token_id, device=device, with_grad=False)
    old_token_lp = old_token_lp.detach()
    pre_logprob_values = [float(value) for value in (old_token_lp * old_mask).sum(dim=1).cpu().tolist()]

    advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    batch_size = len(samples)
    step_size = int(minibatch_size) if minibatch_size else batch_size

    losses, policy_losses, kl_values = [], [], []
    clipped_tokens, counted_tokens, ratio_max = 0.0, 0.0, 0.0
    for _ in range(max(1, int(num_policy_epochs))):
        for start in range(0, batch_size, step_size):
            stop = min(start + step_size, batch_size)
            minibatch = samples[start:stop]
            cur_lp, mask = token_logprobs(model, minibatch, tokenizer.pad_token_id, device=device, with_grad=True)
            # The minibatch is padded to its own max completion length, never wider than the
            # full-batch tensor; completion tokens are left-aligned, so a head slice lines up.
            old_lp = old_token_lp[start:stop, : cur_lp.shape[1]]
            mask_f = mask.float()
            denom = mask_f.sum().clamp_min(1.0)

            ratio = torch.exp((cur_lp - old_lp).float())  # fp32: exp() overflows in bf16
            advantage = advantage_tensor[start:stop].unsqueeze(1)
            objective = torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage)
            policy_loss = -(objective * mask_f).sum() / denom

            kl_value = 0.0
            loss = policy_loss
            if kl_coef > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref_lp, _ = token_logprobs(model, minibatch, tokenizer.pad_token_id, device=device, with_grad=False)
                log_ratio = (ref_lp - cur_lp).float()
                per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
                kl = (per_token_kl * mask_f).sum() / denom
                loss = policy_loss + float(kl_coef) * kl
                kl_value = float(kl.detach().cpu())

            _optimizer_step(model, optimizer, loss, max_grad_norm)

            with torch.no_grad():
                outside = ((ratio - 1.0).abs() > clip_range).float() * mask_f
                clipped_tokens += float(outside.sum().cpu())
                counted_tokens += float(mask_f.sum().cpu())
                ratio_max = max(ratio_max, float((ratio * mask_f).max().cpu()))
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            kl_values.append(kl_value)

    summary = {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "kl": float(np.mean(kl_values)),
        "clip_fraction": clipped_tokens / max(counted_tokens, 1.0),
        "ratio_max": ratio_max,
        "num_minibatch_steps": len(losses),
    }
    return summary, pre_logprob_values


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
    kl_coef: float = 0.0,
    clip_range: float | None = None,
    minibatch_size: int | None = None,
    num_policy_epochs: int = 1,
) -> tuple[dict, list[RolloutSample]]:
    samples = sample_rollouts(
        model,
        tokenizer,
        examples,
        generation_cfg,
        completions_per_prompt=group_size,
        device=device,
    )
    rewards = np.asarray([sample.traits.reward for sample in samples], dtype=float).reshape(len(examples), group_size)
    advantages = compute_group_advantages(rewards, eps=eps).reshape(-1)

    if clip_range is None:
        update_summary, pre_logprob_values = _vanilla_update(
            model, tokenizer, optimizer, samples, advantages, max_grad_norm, device, kl_coef
        )
    else:
        update_summary, pre_logprob_values = _clipped_update(
            model,
            tokenizer,
            optimizer,
            samples,
            advantages,
            max_grad_norm,
            device,
            kl_coef,
            float(clip_range),
            minibatch_size,
            num_policy_epochs,
        )

    train_summary = summarize_trait_metrics(sample.traits for sample in samples)
    train_summary.update(update_summary)
    train_summary.update(
        {
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
        )
        results[name] = summarize_trait_metrics(sample.traits for sample in samples)
    return results
