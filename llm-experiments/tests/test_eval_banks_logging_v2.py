import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.eval_banks import build_eval_banks, select_bank_examples
from src.generation import GeneratedCompletion
from src.grpo import RolloutSample
from src.price_logging import write_price_examples
from src.traits import evaluate_completion_traits


def _config(seed=123):
    return {
        "RunConfig": {"seed": seed},
        "DataConfig": {
            "difficulty": "easy",
            "train_hint_correct_probability": 0.9,
            "eval_distributions": {
                "train_high_hint": {"hint_correct_probability": 0.9, "has_hint": True},
                "eval_balanced_hint": {"hint_correct_probability": 0.5, "has_hint": True},
                "eval_wrong_hint": {"hint_correct_probability": 0.0, "has_hint": True},
                "eval_no_hint": {"has_hint": False},
            },
            "train_hint_phrases": ["hint {choice}"],
            "eval_hint_phrases": ["eval {choice}"],
        },
        "ObservedEvalConfig": {"prompt_bank_size_per_distribution": 8},
        "PriceConfig": {"prompt_bank_size_per_distribution": 8},
    }


def test_build_eval_banks_is_deterministic_and_no_hint_bank_has_no_hints():
    first = build_eval_banks(_config())
    second = build_eval_banks(_config())

    assert [ex.to_json_dict() for ex in first["eval_wrong_hint"]] == [ex.to_json_dict() for ex in second["eval_wrong_hint"]]
    assert all(ex.user_hint is None and ex.hint_is_correct is None for ex in first["eval_no_hint"])


def test_eval_bank_size_uses_larger_observed_or_price_bank_request():
    cfg = _config()
    cfg["ObservedEvalConfig"]["prompt_bank_size_per_distribution"] = 4
    cfg["PriceConfig"]["prompt_bank_size_per_distribution"] = 11

    banks = build_eval_banks(cfg)

    assert len(banks["eval_wrong_hint"]) == 11


def test_select_bank_examples_is_fixed_for_observed_and_step_deterministic_for_price():
    bank = build_eval_banks(_config())["eval_wrong_hint"]

    observed_1 = select_bank_examples(bank, n=4, step=0, seed=1, fixed_across_steps=True)
    observed_2 = select_bank_examples(bank, n=4, step=10, seed=1, fixed_across_steps=True)
    price_1 = select_bank_examples(bank, n=4, step=5, seed=1, fixed_across_steps=False)
    price_2 = select_bank_examples(bank, n=4, step=5, seed=1, fixed_across_steps=False)

    assert [ex.problem_id for ex in observed_1] == [ex.problem_id for ex in observed_2]
    assert [ex.problem_id for ex in price_1] == [ex.problem_id for ex in price_2]


def test_price_example_logging_writes_required_keys_and_consistent_omega(tmp_path: Path):
    example = build_eval_banks(_config())["eval_wrong_hint"][0]
    generated = GeneratedCompletion(
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        completion_text="<answer>A</answer>",
        stop_reason="answer_stop",
        stopped_on_answer_tag=True,
    )
    sample = RolloutSample(
        example=example,
        generated=generated,
        traits=evaluate_completion_traits(generated.completion_text, example, completion_token_length=2),
        pre_logprob=-2.0,
        post_logprob=-1.0,
    )

    paths = write_price_examples(
        tmp_path,
        step=5,
        distribution="eval_wrong_hint",
        samples=[sample],
        logging_cfg={"enabled": True, "every": 5, "max_examples_per_distribution": 1, "include_token_ids": True},
    )

    row = json.loads(paths[0].read_text().splitlines()[0])
    for key in [
        "step",
        "distribution",
        "problem_id",
        "prompt_text",
        "formatted_prompt_ids",
        "completion_ids",
        "stop_reason",
        "parsed_choice",
        "strict_valid",
        "reward",
        "output_agreement",
        "pre_sequence_logprob",
        "post_sequence_logprob",
        "omega",
    ]:
        assert key in row
    assert np.isclose(row["omega"], np.exp(1.0))
