"""Solve-rate comparison across prompt distributions -- the behavioural distribution-shift
test (#3). Generates from the FROZEN base policy (pre-GRPO, no LoRA) on N prompts from each
source, scores with the same Answer:N verifier used for the GRPO reward, and reports the
accuracy gap with a bootstrap CI. If the model solves GSM8K and SVAMP at similar rates then
-- despite surface length differences -- they are effectively the same distribution to the
model, and cross-distribution Price transfer (Q2) is close to trivial; a real gap means the
shift is behaviourally meaningful.

  uv run --no-sync python -m experiments.run_solverate_compare \
      --config experiments/configs/config_real_lr1e-4.yaml --n-prompts 128 --sources eval svamp
"""
import argparse, json, logging
from pathlib import Path
import numpy as np, yaml
from src.model import load_policy
from src.data import load_prompts, build_prompt
from src.grpo import sample_completions
from src.verifier import reward, extract_answer

LOGGER = logging.getLogger(__name__)


def _eval_source(policy, cfg, gen, source, n_prompts, G):
    # Per-prompt solve rate = mean reward over G samples; accuracy = mean over prompts.
    prompts = load_prompts(cfg, source)[:n_prompts]
    rates, spot = [], []
    for i, ex in enumerate(prompts):
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        comps = sample_completions(policy, pid, G, gen)
        texts = [policy.tokenizer.decode(c, skip_special_tokens=True) for c in comps]
        rates.append(sum(reward(t, ex["gold"]) for t in texts) / len(texts))
        if i < 5:
            spot.append({"q": ex["question"][:120], "pred": extract_answer(texts[0]),
                         "gold": ex["gold"], "reward": rates[-1]})
        if (i + 1) % 25 == 0:
            LOGGER.info(f"[{source}] {i + 1}/{len(prompts)} running acc={np.mean(rates):.3f}")
    return np.array(rates), spot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--sources", nargs="+", default=["eval", "svamp"])
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    # Match the capability-check methodology (its temperature), frozen base policy (no LoRA).
    gen = dict(cfg["GenConfig"]); gen["temperature"] = cfg["CapabilityConfig"]["temperature"]
    G = cfg["GRPOConfig"]["G"]

    policy = load_policy(cfg, "cuda")
    rng = np.random.default_rng(cfg["ModelConfig"]["seed"] + 4242)

    results = {}
    for src in args.sources:
        rates, spot = _eval_source(policy, cfg, gen, src, args.n_prompts, G)
        results[src] = {"rates": rates, "spot": spot}
        LOGGER.info(f"[{src}] accuracy={rates.mean():.4f} (n={len(rates)}, G={G})")

    # bootstrap over prompts (independent resample per source); report acc + pairwise gap CIs.
    B = 10000
    def boot(a): return np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(B)])
    boots = {s: boot(r["rates"]) for s, r in results.items()}

    report = {"n_prompts": args.n_prompts, "G": G, "temperature": gen["temperature"],
              "per_source": {}, "gaps": {}}
    for s, r in results.items():
        rt = r["rates"]; bs = boots[s]
        report["per_source"][s] = {
            "accuracy": round(float(rt.mean()), 4), "sd_over_prompts": round(float(rt.std(ddof=1)), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "n": int(len(rt)), "spot_check": r["spot"]}
    srcs = args.sources
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            g = boots[srcs[i]] - boots[srcs[j]]
            key = f"{srcs[i]}_minus_{srcs[j]}"
            report["gaps"][key] = {
                "delta": round(float(results[srcs[i]]["rates"].mean()
                                     - results[srcs[j]]["rates"].mean()), 4),
                "ci95": [round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)],
                "p_gap_gt_0": round(float((g > 0).mean()), 4)}

    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"solverate_compare{args.out_suffix}.json"
    json.dump(report, open(path, "w"), indent=2)
    LOGGER.info("=== solve-rate comparison ===")
    for s in srcs:
        ps = report["per_source"][s]
        LOGGER.info(f"  {s:>6}: acc={ps['accuracy']:.3f}  ci95={ps['ci95']}  (n={ps['n']})")
    for k, v in report["gaps"].items():
        LOGGER.info(f"  gap {k}: Δ={v['delta']:+.3f}  ci95={v['ci95']}  P(Δ>0)={v['p_gap_gt_0']}")
    LOGGER.info(f"done -> {path}")


if __name__ == "__main__":
    main()
