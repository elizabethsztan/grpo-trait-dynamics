from .parsing import parse_answer
from .parsing import parse_answer_with_metadata


def correctness_reward(completion_text: str, gold_choice: str) -> float:
    parsed = parse_answer_with_metadata(completion_text)
    return 1.0 if parsed.valid and parsed.choice == gold_choice else 0.0
