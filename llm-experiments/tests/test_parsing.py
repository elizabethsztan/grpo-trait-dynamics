from src.parsing import parse_answer, parse_answer_with_metadata


def test_parse_answer_accepts_strict_answer_tags():
    assert parse_answer("<answer>A</answer>") == "A"
    assert parse_answer("<answer> b </answer>") == "B"


def test_parse_answer_rejects_untagged_or_invalid_choices():
    assert parse_answer("Answer: B") is None
    assert parse_answer("<answer>E</answer>") is None


def test_multiple_answer_tags_returns_first_and_sets_flag():
    parsed = parse_answer_with_metadata("<answer>C</answer> then <answer>D</answer>")

    assert parsed.choice == "C"
    assert parsed.multiple_answer_tags is True
