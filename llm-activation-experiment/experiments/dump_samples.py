"""Dump a few real (prompt, action) rollouts from the base policy and a trained
checkpoint, with reward, for documentation. Not part of the pipeline."""
import argparse, json
from pathlib import Path
import yaml
from src.model import load_policy
from src.data import load_gsm8k_splits, build_prompt
from src.grpo import sample_completions_batch
from src.verifier import reward as reward_fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--n_prompts", type=int, default=2)
    ap.add_argument("--g", type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    gen = cfg["GenConfig"]
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]

    policy = load_policy(cfg, "cuda")
    eval_ex = load_gsm8k_splits(cfg)["eval"]
    ckpts = sorted((out / "checkpoints").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    stages = [("base (step 0, no LoRA)", ckpts[0]), (f"trained ({ckpts[-1].name})", ckpts[-1])]

    dump = []
    for label, ckpt in stages:
        policy.load_lora(ckpt)
        block = {"stage": label, "checkpoint": ckpt.name, "prompts": []}
        for ex in eval_ex[: args.n_prompts]:
            prompt = build_prompt(ex["question"], n_shots=gen["n_shots"])
            pid = policy.tokenizer(prompt, return_tensors="pt").input_ids[0]
            comps = sample_completions_batch(policy, [pid] * args.g, gen)
            acts = []
            for c in comps:
                text = policy.tokenizer.decode(c, skip_special_tokens=True)
                acts.append({"action": text, "reward": reward_fn(text, ex["gold"])})
            block["prompts"].append({"question": ex["question"], "gold": ex["gold"],
                                     "prompt": prompt, "actions": acts})
        dump.append(block)
    (out / "sample_rollouts.json").write_text(json.dumps(dump, indent=2))
    print(f"wrote {out}/sample_rollouts.json")


if __name__ == "__main__":
    main()
