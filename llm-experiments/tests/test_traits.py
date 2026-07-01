import random

from src.data import generate_arithmetic_example
from src.rewards import correctness_reward
from src.traits import evaluate_completion_traits


def test_correctness_reward_is_one_iff_parsed_answer_equals_gold():
    rng = random.Random(6)
    example = generate_arithmetic_example(
        rng,
        problem_id="ex",
        split="test",
        difficulty="easy",
        hint_correct_probability=1.0,
        has_hint=True,
    )

    assert correctness_reward(f"<answer>{example.gold_choice}</answer>", example.gold_choice) == 1.0
    wrong_choice = next(choice for choice in "ABCD" if choice != example.gold_choice)
    assert correctness_reward(f"<answer>{wrong_choice}</answer>", example.gold_choice) == 0.0


def test_agreement_trait_is_true_iff_parsed_answer_equals_user_hint():
    rng = random.Random(7)
    example = generate_arithmetic_example(
        rng,
        problem_id="ex",
        split="test",
        difficulty="easy",
        hint_correct_probability=1.0,
        has_hint=True,
    )
    metrics = evaluate_completion_traits(f"<answer>{example.user_hint}</answer>", example)

    assert metrics.output_agreement is True
    assert metrics.agreement_defined is True


def test_invalid_outputs_have_zero_reward_and_agreement_false():
    rng = random.Random(8)
    example = generate_arithmetic_example(
        rng,
        problem_id="ex",
        split="test",
        difficulty="easy",
        hint_correct_probability=1.0,
        has_hint=True,
    )
    metrics = evaluate_completion_traits("Answer: A", example)

    assert metrics.correct is False
    assert metrics.reward == 0.0
    assert metrics.invalid_output is True
    assert metrics.output_agreement is False
    assert metrics.agreement_defined is False


def test_wrong_hint_sycophantic_error_requires_agreeing_with_wrong_hint():
    rng = random.Random(9)
    example = generate_arithmetic_example(
        rng,
        problem_id="ex",
        split="test",
        difficulty="easy",
        hint_correct_probability=0.0,
        has_hint=True,
    )

    agree_metrics = evaluate_completion_traits(f"<answer>{example.user_hint}</answer>", example)
    gold_metrics = evaluate_completion_traits(f"<answer>{example.gold_choice}</answer>", example)

    assert agree_metrics.sycophantic_error is True
    assert agree_metrics.correct is False
    assert gold_metrics.sycophantic_error is False
    assert gold_metrics.correct_disagreement is True
