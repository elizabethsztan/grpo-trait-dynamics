from src.parsing import parse_answer, parse_answer_with_metadata
from src.traits import evaluate_completion_traits
from src.data import MCArithmeticExample


def _example(gold="A", hint="A"):
    return MCArithmeticExample(
        problem_id="ex",
        split="test",
        problem_text="What is 1+1?",
        gold_value=2,
        options={"A": 2, "B": 3, "C": 4, "D": 5},
        gold_choice=gold,
        user_hint=hint,
        hint_is_correct=hint == gold if hint is not None else None,
        hint_phrase=None,
        prompt_text="prompt",
    )


def test_parse_answer_accepts_only_one_exact_answer_tag_after_outer_trim():
    cases = [
        ("<answer>A</answer>", "A"),
        ("<answer>b</answer>", "B"),
        (" <answer>C</answer> ", "C"),
    ]

    for text, expected in cases:
        parsed = parse_answer_with_metadata(text)
        assert parsed.valid is True
        assert parsed.choice == expected
        assert parsed.multiple_answer_tags is False
        assert parsed.extra_text is False
        assert parsed.malformed_answer_tag is False
        assert parse_answer(text) == expected


def test_parse_answer_rejects_extra_text_multiple_tags_and_malformed_tags():
    cases = [
        ("The answer is <answer>A</answer>", {"extra_text": True}),
        ("<answer>A</answer>.", {"extra_text": True}),
        ("<answer>A</answer><answer>B</answer>", {"multiple_answer_tags": True}),
        ("<answer>E</answer>", {"malformed_answer_tag": True}),
        ("<answer>A", {"malformed_answer_tag": True}),
        ("A", {}),
    ]

    for text, flags in cases:
        parsed = parse_answer_with_metadata(text)
        assert parsed.valid is False
        assert parsed.choice is None
        assert parse_answer(text) is None
        for flag, expected in flags.items():
            assert getattr(parsed, flag) is expected


def test_multiple_answer_tags_receive_zero_reward_even_if_first_is_gold():
    metrics = evaluate_completion_traits("<answer>A</answer><answer>B</answer>", _example(gold="A", hint="A"))

    assert metrics.parsed_choice is None
    assert metrics.strict_valid is False
    assert metrics.multiple_answer_tags is True
    assert metrics.reward == 0.0
    assert metrics.correct is False
    assert metrics.output_agreement is False
