import argparse, json, logging
from pathlib import Path
import yaml
from src.model import load_policy
from src.data import load_gsm8k_splits, build_prompt
from src.grpo import sample_completions
from src.verifier import reward, extract_answer

LOGGER = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    gen = dict(cfg["GenConfig"]); gen["temperature"] = cfg["CapabilityConfig"]["temperature"]

    policy = load_policy(cfg, "cuda")
    splits = load_gsm8k_splits(cfg)
    prompts = splits["feat"][: cfg["CapabilityConfig"]["n_prompts"]]
    G = cfg["GRPOConfig"]["G"]

    per_prompt_rates, spot = [], []
    for i, ex in enumerate(prompts):
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        comps = sample_completions(policy, pid, G, gen)
        texts = [policy.tokenizer.decode(c, skip_special_tokens=True) for c in comps]
        rates = [reward(t, ex["gold"]) for t in texts]
        per_prompt_rates.append(sum(rates) / len(rates))
        if i < 5:
            spot.append({"completion": texts[0][-200:], "pred": extract_answer(texts[0]),
                         "gold": ex["gold"], "reward": rates[0]})

    lo, hi = cfg["CapabilityConfig"]["target_success_band"]
    in_band = sum(lo <= p_ <= hi for p_ in per_prompt_rates) / len(per_prompt_rates)
    report = {
        "accuracy": round(sum(per_prompt_rates) / len(per_prompt_rates), 4),
        "frac_prompts_in_band": round(in_band, 4),
        "success_rate_hist": [round(p_, 4) for p_ in per_prompt_rates],
        "temperature": gen["temperature"], "max_new_tokens": gen["max_new_tokens"],
        "spot_check": spot,
    }
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / "capability_report.json", "w"), indent=2)
    LOGGER.info(f"accuracy={report['accuracy']} in_band={report['frac_prompts_in_band']} -> {out}")


if __name__ == "__main__":
    main()
