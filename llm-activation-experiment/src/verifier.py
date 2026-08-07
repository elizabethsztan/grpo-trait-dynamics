import re

# Completions are prompted to end with "Answer: N". Take the LAST such match
# (the model may echo the pattern mid-reasoning). Fallback: last number.
_ANSWER_RE = re.compile(r"Answer:\s*(-?\$?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(num_str):
    return num_str.replace(",", "").replace("$", "").rstrip(".").strip()


def extract_answer(completion):
    # Take the last "Answer: N" occurrence for robustness; else the last number.
    matches = _ANSWER_RE.findall(completion)
    if matches:
        return _normalize(matches[-1])
    nums = _NUM_RE.findall(completion)
    return _normalize(nums[-1]) if nums else None


def reward(completion, gold):
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    gold_n = _normalize(gold)
    try:
        return 1.0 if abs(float(pred) - float(gold_n)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if pred == gold_n else 0.0
