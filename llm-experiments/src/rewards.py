from .parsing import parse_answer


def correctness_reward(completion_text: str, gold_choice: str) -> float:
    return 1.0 if parse_answer(completion_text) == gold_choice else 0.0
