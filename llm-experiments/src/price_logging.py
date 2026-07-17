from __future__ import annotations

import json
import math
from pathlib import Path


REQUIRED_PRICE_EXAMPLE_KEYS = [
    "step",
    "distribution",
    "problem_id",
    "prompt_text",
    "formatted_prompt_ids",
    "gold_choice",
    "user_hint",
    "hint_is_correct",
    "completion_text",
    "completion_ids",
    "stop_reason",
    "stopped_on_answer_tag",
    "parsed_choice",
    "strict_valid",
    "reward",
    "output_agreement",
    "activation_agreement",
    "pre_sequence_logprob",
    "post_sequence_logprob",
    "omega",
]


def write_price_examples(run_dir: str | Path, step: int, distribution: str, samples, logging_cfg: dict) -> list[Path]:
    if not logging_cfg.get("enabled", False):
        return []
    every = int(logging_cfg.get("every", 5))
    if every <= 0 or step % every != 0:
        return []
    limit = int(logging_cfg.get("max_examples_per_distribution", 32))
    include_token_ids = bool(logging_cfg.get("include_token_ids", True))
    path = Path(run_dir) / "price_examples" / f"step_{step:04d}_{distribution}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sample in samples[:limit]:
            pre = float(sample.pre_logprob) if sample.pre_logprob is not None else None
            post = float(sample.post_logprob) if sample.post_logprob is not None else None
            omega = math.exp(post - pre) if pre is not None and post is not None else None
            row = {
                "step": int(step),
                "distribution": distribution,
                "problem_id": sample.example.problem_id,
                "prompt_text": sample.example.prompt_text,
                "formatted_prompt_ids": sample.generated.prompt_ids if include_token_ids else None,
                "gold_choice": sample.example.gold_choice,
                "user_hint": sample.example.user_hint,
                "hint_is_correct": sample.example.hint_is_correct,
                "completion_text": sample.generated.completion_text,
                "completion_ids": sample.generated.completion_ids if include_token_ids else None,
                "stop_reason": sample.generated.stop_reason,
                "stopped_on_answer_tag": sample.generated.stopped_on_answer_tag,
                "parsed_choice": sample.traits.parsed_choice,
                "strict_valid": sample.traits.strict_valid,
                "reward": sample.traits.reward,
                "output_agreement": sample.traits.output_agreement,
                "activation_agreement": sample.traits.activation_agreement,
                "pre_sequence_logprob": pre,
                "post_sequence_logprob": post,
                "omega": omega,
            }
            f.write(json.dumps(row) + "\n")
    return [path]
