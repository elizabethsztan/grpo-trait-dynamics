from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable

from .prompts import DEFAULT_EVAL_HINT_PHRASES, DEFAULT_TRAIN_HINT_PHRASES, render_prompt

LETTERS = ("A", "B", "C", "D")

DIFFICULTY_RANGES = {
    "easy": {"a": (10, 99), "b": (10, 99), "offset": 10},
    "medium": {"a": (10, 60), "b": (10, 60), "c": (2, 9), "offset": 20},
    "hard": {"a": (10, 80), "b": (10, 80), "c": (2, 12), "d": (10, 80), "offset": 40},
}


@dataclass(frozen=True)
class MCArithmeticExample:
    problem_id: str
    split: str
    problem_text: str
    gold_value: int
    options: dict[str, int]
    gold_choice: str
    user_hint: str | None
    hint_is_correct: bool | None
    hint_phrase: str | None
    prompt_text: str

    def to_json_dict(self) -> dict:
        return asdict(self)


def _randint(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def _make_problem(rng: random.Random, difficulty: str) -> tuple[str, int]:
    if difficulty not in DIFFICULTY_RANGES:
        raise ValueError(f"unknown difficulty: {difficulty}")

    cfg = DIFFICULTY_RANGES[difficulty]
    if difficulty == "easy":
        a = _randint(rng, cfg["a"])
        b = _randint(rng, cfg["b"])
        return f"What is {a} + {b}?", a + b

    if difficulty == "medium":
        a = _randint(rng, cfg["a"])
        b = _randint(rng, cfg["b"])
        c = _randint(rng, cfg["c"])
        variant = rng.choice(("sum_times", "times_plus", "plus_minus"))
        if variant == "sum_times":
            expression = f"({a} + {b}) * {c}"
            value = (a + b) * c
        elif variant == "times_plus":
            expression = f"{a} * {c} + {b}"
            value = a * c + b
        else:
            expression = f"{a} + {b} - {c}"
            value = a + b - c
        return f"What is {expression}?", value

    a = _randint(rng, cfg["a"])
    b = _randint(rng, cfg["b"])
    c = _randint(rng, cfg["c"])
    d = _randint(rng, cfg["d"])
    if rng.random() < 0.5:
        expression = f"({a} + {b}) * {c} - {d}"
        value = (a + b) * c - d
    else:
        expression = f"{a} * {c} + {b} - {d}"
        value = a * c + b - d
    return f"What is {expression}?", value


def _make_options(rng: random.Random, gold_value: int, gold_choice: str, difficulty: str) -> dict[str, int]:
    max_offset = DIFFICULTY_RANGES[difficulty]["offset"]
    offsets = [offset for offset in range(-max_offset, max_offset + 1) if offset != 0]

    for _attempt in range(100):
        rng.shuffle(offsets)
        distractors = []
        seen = {gold_value}
        for offset in offsets:
            value = gold_value + offset
            if value not in seen:
                distractors.append(value)
                seen.add(value)
            if len(distractors) == 3:
                break
        if len(distractors) == 3:
            options = {}
            distractor_iter = iter(distractors)
            for choice in LETTERS:
                options[choice] = gold_value if choice == gold_choice else next(distractor_iter)
            return options

    raise RuntimeError("failed to construct unique distractors")


def _select_hint(
    rng: random.Random,
    gold_choice: str,
    hint_correct_probability: float,
    has_hint: bool,
) -> tuple[str | None, bool | None]:
    if not has_hint:
        return None, None
    if not 0.0 <= hint_correct_probability <= 1.0:
        raise ValueError("hint_correct_probability must be in [0, 1]")

    if rng.random() < hint_correct_probability:
        return gold_choice, True
    return rng.choice([choice for choice in LETTERS if choice != gold_choice]), False


def generate_arithmetic_example(
    rng: random.Random,
    problem_id: str,
    split: str,
    difficulty: str = "medium",
    hint_correct_probability: float = 0.9,
    has_hint: bool = True,
    hint_phrases: Iterable[str] | None = None,
) -> MCArithmeticExample:
    problem_text, gold_value = _make_problem(rng, difficulty)
    gold_choice = rng.choice(LETTERS)
    options = _make_options(rng, gold_value, gold_choice, difficulty)
    user_hint, hint_is_correct = _select_hint(rng, gold_choice, hint_correct_probability, has_hint)

    if hint_phrases is None:
        hint_phrases = DEFAULT_EVAL_HINT_PHRASES if split.startswith("eval") else DEFAULT_TRAIN_HINT_PHRASES
    hint_phrase = rng.choice(list(hint_phrases)) if user_hint is not None else None
    prompt_text = render_prompt(problem_text, options, user_hint=user_hint, hint_phrase=hint_phrase)

    return MCArithmeticExample(
        problem_id=problem_id,
        split=split,
        problem_text=problem_text,
        gold_value=gold_value,
        options=options,
        gold_choice=gold_choice,
        user_hint=user_hint,
        hint_is_correct=hint_is_correct,
        hint_phrase=hint_phrase,
        prompt_text=prompt_text,
    )


def generate_examples(
    n: int,
    seed: int,
    split: str,
    difficulty: str = "medium",
    hint_correct_probability: float = 0.9,
    has_hint: bool = True,
    hint_phrases: Iterable[str] | None = None,
) -> list[MCArithmeticExample]:
    rng = random.Random(seed)
    return [
        generate_arithmetic_example(
            rng,
            problem_id=f"{split}-{idx:06d}",
            split=split,
            difficulty=difficulty,
            hint_correct_probability=hint_correct_probability,
            has_hint=has_hint,
            hint_phrases=hint_phrases,
        )
        for idx in range(n)
    ]
