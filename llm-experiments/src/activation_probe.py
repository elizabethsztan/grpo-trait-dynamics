from __future__ import annotations

from dataclasses import dataclass
import logging
import random

from .data import LETTERS, generate_arithmetic_example
from .prompts import DEFAULT_TRAIN_HINT_PHRASES, render_prompt
from .prompts import encode_prompt_for_generation

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbePair:
    agree_prompt: str
    disagree_prompt: str
    completion: str
    completion_choice: str
    contrast_hint: str
    disagree_completion: str | None = None


@dataclass
class ActivationAgreementProbe:
    hook_layer: int
    vector: object
    score_mean: float
    score_std: float
    pooling: str = "answer_tag_tokens"
    stop_token_ids: tuple[int, ...] = ()

    def score_pooled(self, pooled_hidden):
        return ((pooled_hidden @ self.vector) - self.score_mean) / self.score_std

    def score_texts(self, model, tokenizer, prompt_texts, completion_texts, device):
        pooled = pool_completion_hidden(model, tokenizer, prompt_texts, completion_texts, self.hook_layer, device, pooling=self.pooling)
        return self.score_pooled(pooled)

    def score_samples(self, model, samples, device):
        pooled = pool_sample_hidden(model, samples, self.hook_layer, device, pooling=self.pooling, stop_token_ids=self.stop_token_ids)
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


def make_completion_counterfactual_pairs(
    n: int,
    seed: int,
    difficulty: str = "medium",
    hint_phrases=None,
) -> list[ProbePair]:
    rng = random.Random(seed)
    phrases = list(hint_phrases or DEFAULT_TRAIN_HINT_PHRASES)
    pairs = []
    for idx in range(n):
        hint = LETTERS[idx % len(LETTERS)]
        disagree = rng.choice([choice for choice in LETTERS if choice != hint])
        example = generate_arithmetic_example(
            rng,
            problem_id=f"probe-completion-{idx:06d}",
            split="probe",
            difficulty=difficulty,
            hint_correct_probability=0.5,
            has_hint=True,
            hint_phrases=phrases,
        )
        phrase = phrases[idx % len(phrases)]
        prompt = render_prompt(
            example.problem_text,
            example.options,
            user_hint=hint,
            hint_phrase=phrase,
        )
        pairs.append(
            ProbePair(
                agree_prompt=prompt,
                disagree_prompt=prompt,
                completion=f"<answer>{hint}</answer>",
                disagree_completion=f"<answer>{disagree}</answer>",
                completion_choice=hint,
                contrast_hint=disagree,
            )
        )
    rng.shuffle(pairs)
    return pairs


def _stop_suffix_index(token_ids: list[int], stop_token_ids: tuple[int, ...]) -> int | None:
    if not stop_token_ids:
        return None
    stop = list(stop_token_ids)
    width = len(stop)
    for end in range(width, len(token_ids) + 1):
        if token_ids[end - width : end] == stop:
            return end
    return None


def _pool_stop_for_completion(completion_ids: list[int], pooling: str, stop_token_ids: tuple[int, ...]) -> int:
    if pooling in ("whole_completion_tokens", "mean_completion_tokens"):
        return len(completion_ids)
    if pooling in ("answer_tag_tokens", "answer_letter_tokens"):
        stop = _stop_suffix_index(completion_ids, stop_token_ids)
        return stop if stop is not None else len(completion_ids)
    raise ValueError(f"unsupported activation pooling mode: {pooling}")


def _pool_from_token_ids(model, token_pairs, hook_layer: int, device, pooling: str = "answer_tag_tokens", stop_token_ids=()):
    import torch

    pooled = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        stop_token_ids = tuple(int(token_id) for token_id in stop_token_ids)
        for prompt_ids, completion_ids in token_pairs:
            if not completion_ids:
                raise ValueError("activation scoring requires at least one completion token")
            ids = prompt_ids + completion_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hook_hidden = outputs.hidden_states[hook_layer + 1][0]
            start = len(prompt_ids)
            stop = start + _pool_stop_for_completion(list(completion_ids), pooling=pooling, stop_token_ids=stop_token_ids)
            pooled.append(hook_hidden[start:stop].mean(dim=0))
    if was_training:
        model.train()
    return torch.stack(pooled)


def pool_completion_hidden(model, tokenizer, prompt_texts, completion_texts, hook_layer: int, device, pooling: str = "answer_tag_tokens"):
    token_pairs = []
    stop_token_ids = tuple(tokenizer.encode("</answer>", add_special_tokens=False)) if pooling in ("answer_tag_tokens", "answer_letter_tokens") else ()
    for prompt_text, completion_text in zip(prompt_texts, completion_texts):
        prompt_ids = encode_prompt_for_generation(tokenizer, prompt_text, use_chat_template=True)
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        token_pairs.append((prompt_ids, completion_ids))
    return _pool_from_token_ids(model, token_pairs, hook_layer, device, pooling=pooling, stop_token_ids=stop_token_ids)


def pool_sample_hidden(model, samples, hook_layer: int, device, pooling: str = "answer_tag_tokens", stop_token_ids=()):
    token_pairs = [(list(sample.generated.prompt_ids), list(sample.generated.completion_ids)) for sample in samples]
    return _pool_from_token_ids(model, token_pairs, hook_layer, device, pooling=pooling, stop_token_ids=tuple(stop_token_ids))


def build_activation_probe(model, tokenizer, cfg: dict, data_cfg: dict, seed: int, device):
    import torch

    hook_layer = int(cfg.get("hook_layer", cfg.get("selected_hook_layer", 12)))
    phrases = data_cfg.get("train_hint_phrases")
    construction = cfg.get("construction", "completion_counterfactual")
    pooling = cfg.get("pooling", "answer_tag_tokens")
    pair_builder = make_completion_counterfactual_pairs if construction == "completion_counterfactual" else make_probe_pairs
    vector_pairs = pair_builder(
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
        pooling=pooling,
    )
    disagree = pool_completion_hidden(
        model,
        tokenizer,
        [pair.disagree_prompt for pair in vector_pairs],
        [pair.disagree_completion or pair.completion for pair in vector_pairs],
        hook_layer,
        device,
        pooling=pooling,
    )
    vector = (agree - disagree).mean(dim=0)
    norm = torch.linalg.norm(vector)
    if float(norm) == 0.0:
        raise RuntimeError("activation probe vector has zero norm")
    vector = vector / norm

    norm_pairs = pair_builder(
        int(cfg.get("normalization_pairs", 32)),
        seed=seed + 1,
        difficulty=data_cfg.get("difficulty", "medium"),
        hint_phrases=phrases,
    )
    prompts = [pair.agree_prompt for pair in norm_pairs] + [pair.disagree_prompt for pair in norm_pairs]
    completions = [pair.completion for pair in norm_pairs] + [pair.disagree_completion or pair.completion for pair in norm_pairs]
    raw_scores = pool_completion_hidden(model, tokenizer, prompts, completions, hook_layer, device, pooling=pooling) @ vector
    score_mean = float(raw_scores.mean())
    score_std = float(raw_scores.std(unbiased=False))
    if score_std <= 1e-12:
        LOGGER.warning("activation probe normalization std is numerically zero; using 1.0")
        score_std = 1.0
    return ActivationAgreementProbe(
        hook_layer=hook_layer,
        vector=vector,
        score_mean=score_mean,
        score_std=score_std,
        pooling=pooling,
        stop_token_ids=tuple(tokenizer.encode("</answer>", add_special_tokens=False)),
    )
