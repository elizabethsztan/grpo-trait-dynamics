from __future__ import annotations

from dataclasses import dataclass
import logging
import random

from .data import LETTERS, generate_arithmetic_example
from .prompts import DEFAULT_TRAIN_HINT_PHRASES, render_prompt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbePair:
    agree_prompt: str
    disagree_prompt: str
    completion: str
    completion_choice: str
    contrast_hint: str


@dataclass
class ActivationAgreementProbe:
    hook_layer: int
    vector: object
    score_mean: float
    score_std: float

    def score_pooled(self, pooled_hidden):
        return ((pooled_hidden @ self.vector) - self.score_mean) / self.score_std

    def score_texts(self, model, tokenizer, prompt_texts, completion_texts, device):
        pooled = pool_completion_hidden(model, tokenizer, prompt_texts, completion_texts, self.hook_layer, device)
        return self.score_pooled(pooled)


def make_probe_pairs(
    n: int,
    seed: int,
    difficulty: str = "medium",
    hint_phrases=None,
) -> list[ProbePair]:
    rng = random.Random(seed)
    phrases = list(hint_phrases or DEFAULT_TRAIN_HINT_PHRASES)
    pairs = []
    for idx in range(n):
        completion_choice = LETTERS[idx % len(LETTERS)]
        contrast_hint = rng.choice([choice for choice in LETTERS if choice != completion_choice])
        example = generate_arithmetic_example(
            rng,
            problem_id=f"probe-{idx:06d}",
            split="probe",
            difficulty=difficulty,
            hint_correct_probability=0.5,
            has_hint=True,
            hint_phrases=phrases,
        )
        phrase = phrases[idx % len(phrases)]
        pairs.append(
            ProbePair(
                agree_prompt=render_prompt(
                    example.problem_text,
                    example.options,
                    user_hint=completion_choice,
                    hint_phrase=phrase,
                ),
                disagree_prompt=render_prompt(
                    example.problem_text,
                    example.options,
                    user_hint=contrast_hint,
                    hint_phrase=phrase,
                ),
                completion=f"<answer>{completion_choice}</answer>",
                completion_choice=completion_choice,
                contrast_hint=contrast_hint,
            )
        )
    rng.shuffle(pairs)
    return pairs


def pool_completion_hidden(model, tokenizer, prompt_texts, completion_texts, hook_layer: int, device):
    import torch

    pooled = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for prompt_text, completion_text in zip(prompt_texts, completion_texts):
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
            if not completion_ids:
                completion_ids = [tokenizer.eos_token_id]
            ids = prompt_ids + completion_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hook_hidden = outputs.hidden_states[hook_layer + 1][0]
            start = len(prompt_ids)
            stop = start + len(completion_ids)
            pooled.append(hook_hidden[start:stop].mean(dim=0))
    if was_training:
        model.train()
    return torch.stack(pooled)


def build_activation_probe(model, tokenizer, cfg: dict, data_cfg: dict, seed: int, device):
    import torch

    hook_layer = int(cfg["hook_layer"] if "hook_layer" in cfg else cfg.get("hook_layer", 12))
    phrases = data_cfg.get("train_hint_phrases")
    vector_pairs = make_probe_pairs(
        int(cfg.get("num_probe_pairs", 32)),
        seed=seed,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_phrases=phrases,
    )
    agree = pool_completion_hidden(
        model,
        tokenizer,
        [pair.agree_prompt for pair in vector_pairs],
        [pair.completion for pair in vector_pairs],
        hook_layer,
        device,
    )
    disagree = pool_completion_hidden(
        model,
        tokenizer,
        [pair.disagree_prompt for pair in vector_pairs],
        [pair.completion for pair in vector_pairs],
        hook_layer,
        device,
    )
    vector = (agree - disagree).mean(dim=0)
    norm = torch.linalg.norm(vector)
    if float(norm) == 0.0:
        raise RuntimeError("activation probe vector has zero norm")
    vector = vector / norm

    norm_pairs = make_probe_pairs(
        int(cfg.get("normalization_pairs", 32)),
        seed=seed + 1,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_phrases=phrases,
    )
    prompts = [pair.agree_prompt for pair in norm_pairs] + [pair.disagree_prompt for pair in norm_pairs]
    completions = [pair.completion for pair in norm_pairs] * 2
    raw_scores = pool_completion_hidden(model, tokenizer, prompts, completions, hook_layer, device) @ vector
    score_mean = float(raw_scores.mean())
    score_std = float(raw_scores.std(unbiased=False))
    if score_std <= 1e-12:
        LOGGER.warning("activation probe normalization std is numerically zero; using 1.0")
        score_std = 1.0
    return ActivationAgreementProbe(hook_layer=hook_layer, vector=vector, score_mean=score_mean, score_std=score_std)
