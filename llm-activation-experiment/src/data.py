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


def _load_split(name, seed):
    # openai/gsm8k is the canonical parquet-backed dataset (no remote code needed).
    ds = load_dataset("openai/gsm8k", "main", split=name).shuffle(seed=seed)
    return [{"question": r["question"], "gold": _gold_from_answer(r["answer"])} for r in ds]


def _load_svamp(seed):
    # SVAMP: simpler single-/few-step math word problems from a different source than GSM8K
    # (grade-school family, numeric answers our verifier handles). Used ONLY as an OOD
    # measurement distribution -- never trained on -- so we pool all 1000 problems.
    import random
    rows = []
    for split in ("train", "test"):
        ds = load_dataset("ChilleD/SVAMP", split=split)
        rows += [{"question": r["question_concat"], "gold": str(r["Answer"])} for r in ds]
    random.Random(seed).shuffle(rows)
    return rows


def load_prompts(cfg, source):
    # Dispatch a prompt source to a list of {"question","gold"}. GSM8K splits ('eval','train',
    # 'feat') go through load_gsm8k_splits; 'svamp' is the OOD math distribution for the
    # cross-distribution Price test (measure trait drift on D' with our GSM8K-trained ckpts).
    if source in ("eval", "train", "feat"):
        return load_gsm8k_splits(cfg)[source]
    if source == "svamp":
        return _load_svamp(cfg["ModelConfig"]["seed"])
    raise ValueError(f"unknown prompt source '{source}'")


def load_gsm8k_splits(cfg):
    data_cfg = cfg["DataConfig"]
    seed = cfg["ModelConfig"]["seed"]
    n_train, n_feat, n_eval = data_cfg["n_train"], data_cfg["n_feat"], data_cfg["n_eval"]
    # GSM8K ships only train (7473) and test (1319) -- no official val. To keep the
    # Price eval reviewer-proof, `train` and `feat` are disjoint slices of the official
    # TRAIN split, and `eval` is drawn from the official TEST split (never trained on).
    train_pool = _load_split("train", seed)
    test_pool = _load_split("test", seed)
    assert n_train + n_feat <= len(train_pool), f"n_train+n_feat={n_train + n_feat} > train {len(train_pool)}"
    assert n_eval <= len(test_pool), f"n_eval={n_eval} > test {len(test_pool)}"
    splits = {
        "train": train_pool[:n_train],
        "feat": train_pool[n_train:n_train + n_feat],
        "eval": test_pool[:n_eval],
    }
    LOGGER.info(f"splits: train={len(splits['train'])} feat={len(splits['feat'])} "
                f"eval={len(splits['eval'])} (eval from official test split)")
    return splits
