from __future__ import annotations

import json
import random
from pathlib import Path

from .data import MCArithmeticExample, generate_examples


DEFAULT_BANK_DISTRIBUTIONS = ("train_high_hint", "eval_balanced_hint", "eval_wrong_hint", "eval_no_hint")


def _bank_size(config: dict, distribution: str) -> int:
    observed = config.get("ObservedEvalConfig", {})
    price = config.get("PriceConfig", {})
    observed_size = int(observed.get("prompt_bank_size_per_distribution", 256))
    price_size = int(price.get("prompt_bank_size_per_distribution", observed_size))
    return max(observed_size, price_size)


def build_eval_banks(config: dict) -> dict[str, list[MCArithmeticExample]]:
    run_cfg = config["RunConfig"]
    data_cfg = config["DataConfig"]
    banks = {}
    for idx, name in enumerate(DEFAULT_BANK_DISTRIBUTIONS):
        dist_cfg = data_cfg["eval_distributions"][name]
        banks[name] = generate_examples(
            n=_bank_size(config, name),
            seed=int(run_cfg["seed"]) + 900000 + idx * 1009,
            split=name,
            difficulty=data_cfg.get("difficulty", "medium"),
            hint_correct_probability=float(dist_cfg.get("hint_correct_probability", data_cfg.get("train_hint_correct_probability", 0.9))),
            has_hint=bool(dist_cfg.get("has_hint", True)),
            hint_phrases=data_cfg.get("eval_hint_phrases" if name.startswith("eval") else "train_hint_phrases"),
        )
    return banks


def save_eval_banks(run_dir: str | Path, banks: dict[str, list[MCArithmeticExample]]) -> None:
    bank_dir = Path(run_dir) / "eval_banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in banks.items():
        path = bank_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for example in examples:
                f.write(json.dumps(example.to_json_dict()) + "\n")


def load_eval_banks(run_dir: str | Path) -> dict[str, list[MCArithmeticExample]]:
    bank_dir = Path(run_dir) / "eval_banks"
    banks = {}
    for path in sorted(bank_dir.glob("*.jsonl")):
        examples = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    examples.append(MCArithmeticExample(**json.loads(line)))
        banks[path.stem] = examples
    return banks


def select_bank_examples(
    bank: list[MCArithmeticExample],
    n: int,
    step: int,
    seed: int,
    fixed_across_steps: bool,
) -> list[MCArithmeticExample]:
    if n >= len(bank):
        return list(bank)
    rng_step = 0 if fixed_across_steps else step
    rng = random.Random(int(seed) + rng_step * 104729)
    indices = sorted(rng.sample(range(len(bank)), n))
    return [bank[idx] for idx in indices]
