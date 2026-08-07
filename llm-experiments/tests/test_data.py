import random

from src.data import LETTERS, generate_arithmetic_example


def test_generated_example_has_valid_gold_and_unique_options():
    rng = random.Random(1)

    example = generate_arithmetic_example(
        rng,
        problem_id="ex-1",
        split="test",
        difficulty="medium",
        hint_correct_probability=0.9,
        has_hint=True,
    )

    assert example.gold_choice in LETTERS
    assert set(example.options) == set(LETTERS)
    assert len(set(example.options.values())) == 4
    assert example.options[example.gold_choice] == example.gold_value


def test_hint_correct_probability_one_always_matches_gold_choice():
    rng = random.Random(2)

    for idx in range(50):
        example = generate_arithmetic_example(
            rng,
            problem_id=f"ex-{idx}",
            split="test",
            difficulty="easy",
            hint_correct_probability=1.0,
            has_hint=True,
        )
        assert example.user_hint == example.gold_choice
        assert example.hint_is_correct is True


def test_hint_correct_probability_zero_never_matches_gold_choice():
    rng = random.Random(3)

    for idx in range(50):
        example = generate_arithmetic_example(
            rng,
            problem_id=f"ex-{idx}",
            split="test",
            difficulty="easy",
            hint_correct_probability=0.0,
            has_hint=True,
        )
        assert example.user_hint != example.gold_choice
        assert example.hint_is_correct is False


def test_no_hint_examples_have_null_hint_fields():
    rng = random.Random(4)

    example = generate_arithmetic_example(
        rng,
        problem_id="ex-no-hint",
        split="test",
        difficulty="easy",
        hint_correct_probability=0.9,
        has_hint=False,
    )

    assert example.user_hint is None
    assert example.hint_is_correct is None
    assert example.hint_phrase is None


def test_gold_choices_are_approximately_balanced_over_many_samples():
    rng = random.Random(5)
    counts = dict.fromkeys(LETTERS, 0)

    for idx in range(800):
        example = generate_arithmetic_example(
            rng,
            problem_id=f"ex-{idx}",
            split="test",
            difficulty="medium",
            hint_correct_probability=0.5,
            has_hint=True,
        )
        counts[example.gold_choice] += 1

    expected = 800 / len(LETTERS)
    for count in counts.values():
        assert abs(count - expected) < 60
