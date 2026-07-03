import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores
from src.logprob import sequence_logprob
from src.grpo import sample_completions
from src.data import load_gsm8k_splits, build_prompt

LOGGER = logging.getLogger(__name__)


def _feature_ids(features):
    ids = [e["feature_id"] for e in features["positive"] + features["negative"]]
    ctrl = [e["feature_id"] for e in features["controls"]]
    return ids, ctrl


def _draw(policy, examples, gen, n, rng):
    # x ~ eval prompts, a ~ pi_t(.|x): one completion per draw.
    draws = []
    for _ in range(n):
        ex = examples[int(rng.integers(len(examples)))]
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        cid = sample_completions(policy, pid, 1, gen)[0]
        draws.append((pid, cid))
    return draws


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    pe, gen = cfg["PriceEvalConfig"], cfg["GenConfig"]
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    features = json.load(open(out / "features.json"))
    layer = features["layer_L"]
    feat_ids, ctrl_ids = _feature_ids(features)
    all_ids = torch.tensor(feat_ids + ctrl_ids)

    policy = load_policy(cfg, "cuda")
    # No attach_lora here: the first load_lora(ckpt) wraps the base from the saved
    # adapter config, and subsequent load_lora calls swap adapter weights in place.
    sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
    eval_ex = load_gsm8k_splits(cfg)["eval"][: pe["eval_prompts"]]

    ckpts = sorted((out / "checkpoints").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    rng = np.random.default_rng(cfg["ModelConfig"]["seed"] + 777)
    N_max = max(pe["price_samples"] + [pe["direct_samples"]])

    with open(out / "price_eval.jsonl", "w") as f:
        for t in range(len(ckpts) - 1):
            policy.load_lora(ckpts[t])
            # price draws from pi_t (independent of the gradient rollouts) + trait scores
            price_draws = _draw(policy, eval_ex, gen, N_max, rng)
            s_price = torch.stack([collect_feature_scores(policy, sae, layer, p_, c).cpu()[all_ids]
                                   for p_, c in price_draws])              # (N_max, F)
            logp_t = torch.tensor([sequence_logprob(policy, p_, c) for p_, c in price_draws])
            # direct T_t (high budget)
            direct_draws = _draw(policy, eval_ex, gen, pe["direct_samples"], rng)
            s_dir_t = torch.stack([collect_feature_scores(policy, sae, layer, p_, c).cpu()[all_ids]
                                   for p_, c in direct_draws]).mean(0)     # (F,)

            policy.load_lora(ckpts[t + 1])
            logp_tp1 = torch.tensor([sequence_logprob(policy, p_, c) for p_, c in price_draws])
            direct_draws2 = _draw(policy, eval_ex, gen, pe["direct_samples"], rng)
            s_dir_tp1 = torch.stack([collect_feature_scores(policy, sae, layer, p_, c).cpu()[all_ids]
                                     for p_, c in direct_draws2]).mean(0)
            direct_drift = (s_dir_tp1 - s_dir_t)                            # (F,)

            omega_full = torch.exp(logp_tp1 - logp_t)                       # (N_max,)
            for N in pe["price_samples"]:
                w, s = omega_full[:N], s_price[:N]
                mean_w = w.mean()
                price = (w[:, None] * s).mean(0) - s.mean(0)                # (F,)
                cov = (w[:, None] * s).mean(0) - mean_w * s.mean(0)
                ess = float((w.sum() ** 2) / (w ** 2).sum())
                for fi, fid in enumerate(feat_ids + ctrl_ids):
                    f.write(json.dumps({
                        "step": t, "feature_id": int(fid),
                        "is_control": fid in ctrl_ids, "N": N,
                        "direct_drift": round(float(direct_drift[fi]), 4),
                        "price": round(float(price[fi]), 4),
                        "cov": round(float(cov[fi]), 4),
                        "mean_omega": round(float(mean_w), 4),
                        "omega_var": round(float(w.var()), 4),
                        "omega_max": round(float(w.max()), 4),
                        "ess": round(ess, 2),
                    }) + "\n")
            LOGGER.info(f"step {t}->{t+1} mean_omega={float(omega_full.mean()):.3f} "
                        f"ess@max={float((omega_full.sum()**2)/(omega_full**2).sum()):.1f}")
    LOGGER.info(f"done -> {out}/price_eval.jsonl")

    from src.plotting import plot_from_jsonl
    plot_from_jsonl(out / "price_eval.jsonl", out / "plots")
    LOGGER.info(f"saved plots to {out}/plots")


if __name__ == "__main__":
    main()
