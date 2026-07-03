import logging
from datasets import load_dataset

LOGGER = logging.getLogger(__name__)

# Four fixed worked examples; completions end with the exact "Answer: N" line
# the verifier keys on. Kept short to bound sequence length.
FEW_SHOT = [
    {"q": "Natalia sold 48 clips in April and half as many in May. How many clips did she sell altogether?",
     "a": "In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72.\nAnswer: 72"},
    {"q": "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?",
     "a": "Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes: 50 * 0.2 = 10.\nAnswer: 10"},
    {"q": "Betty needs $100 for a wallet. She has half. Her parents give $15 and grandparents twice that. How much more does she need?",
     "a": "Betty has 100 / 2 = 50. Grandparents give 2 * 15 = 30. She now has 50 + 15 + 30 = 95. She needs 100 - 95 = 5.\nAnswer: 5"},
    {"q": "James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?",
     "a": "Each time he writes 3 * 2 = 6 pages. Twice a week: 6 * 2 = 12 pages. Per year: 12 * 52 = 624.\nAnswer: 624"},
]


def build_prompt(question, shots=FEW_SHOT, n_shots=4):
    blocks = []
    for ex in shots[:n_shots]:
        blocks.append(f"Question: {ex['q']}\n{ex['a']}")
    blocks.append(f"Question: {question}\n")
    return "\n\n".join(blocks)


def _gold_from_answer(answer_field):
    # GSM8K gold answers end with "#### <number>".
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_gsm8k_splits(cfg):
    data_cfg = cfg["DataConfig"]
    seed = cfg["ModelConfig"]["seed"]
    # openai/gsm8k is the canonical parquet-backed dataset (no remote code needed).
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    examples = [{"question": r["question"], "gold": _gold_from_answer(r["answer"])} for r in ds]
    n_train, n_feat, n_eval = data_cfg["n_train"], data_cfg["n_feat"], data_cfg["n_eval"]
    splits = {
        "train": examples[:n_train],
        "feat": examples[n_train:n_train + n_feat],
        "eval": examples[n_train + n_feat:n_train + n_feat + n_eval],
    }
    LOGGER.info(f"splits: train={len(splits['train'])} feat={len(splits['feat'])} eval={len(splits['eval'])}")
    return splits
