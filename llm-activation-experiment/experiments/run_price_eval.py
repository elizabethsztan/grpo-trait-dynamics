import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores_batch
from src.logprob import sequence_logprob_batch
from src.grpo import sample_completions_batch
from src.data import load_gsm8k_splits, build_prompt

LOGGER = logging.getLogger(__name__)


def _feature_ids(features):
    ids = [e["feature_id"] for e in features["positive"] + features["negative"]]
    ctrl = [e["feature_id"] for e in features["controls"]]
    return ids, ctrl


def _chunks(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def _draw(policy, examples, gen, n, rng, bs):
    # x ~ eval prompts, a ~ pi_t(.|x): one completion per draw, generated in batches.
    pids = [policy.tokenizer(build_prompt(examples[int(rng.integers(len(examples)))]["question"],
                                          n_shots=gen["n_shots"]), return_tensors="pt").input_ids[0]
            for _ in range(n)]
    draws = []
    for chunk in _chunks(pids, bs):
        comps = sample_completions_batch(policy, chunk, gen)
        draws.extend(zip(chunk, comps))
    return draws


def _trait_matrix(policy, sae, layer, draws, all_ids, bs):
    # (len(draws), F) trait scores on CPU, batched teacher-forced.
    rows = []
    for chunk in _chunks(draws, bs):
        s = collect_feature_scores_batch(policy, sae, layer, chunk).cpu()   # (b, n_features)
        rows.append(s[:, all_ids])
    return torch.cat(rows, dim=0)


def _logprobs(policy, draws, bs):
    out = []
    for chunk in _chunks(draws, bs):
        out.extend(sequence_logprob_batch(policy, chunk))
    return torch.tensor(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    # --out-suffix: write price_eval<suffix>.jsonl / plots<suffix>/ so repeated Phase-3
    # sweeps on the same checkpoints don't clobber each other.
    # --max-transitions: cap the number of adjacent checkpoint pairs evaluated (0 = all);
    # set to 1 for a cheap single-pair N-convergence diagnostic.
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-transitions", type=int, default=0)
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
    # Price sweep only consumes up to max(price_samples) rollouts; the direct estimate
    # uses its own separate direct_samples draws. (Don't over-draw to direct_samples.)
    N_price = max(pe["price_samples"])
    bs = pe.get("batch_size", 32)

    n_trans = len(ckpts) - 1
    if args.max_transitions > 0:
        n_trans = min(n_trans, args.max_transitions)
    jsonl_path = out / f"price_eval{args.out_suffix}.jsonl"
    with open(jsonl_path, "w") as f:
        for t in range(n_trans):
            policy.load_lora(ckpts[t])
            # price draws from pi_t (independent of the gradient rollouts) + trait scores.
            # The SAME price_draws feed logp_t and logp_tp1 so any batched-forward numerical
            # artifact is common-mode and cancels in omega = exp(logp_tp1 - logp_t).
            price_draws = _draw(policy, eval_ex, gen, N_price, rng, bs)
            s_price = _trait_matrix(policy, sae, layer, price_draws, all_ids, bs)   # (N_price, F)
            logp_t = _logprobs(policy, price_draws, bs)
            # direct T_t (high budget)
            direct_draws = _draw(policy, eval_ex, gen, pe["direct_samples"], rng, bs)
            s_dir_t = _trait_matrix(policy, sae, layer, direct_draws, all_ids, bs).mean(0)   # (F,)

            policy.load_lora(ckpts[t + 1])
            logp_tp1 = _logprobs(policy, price_draws, bs)
            direct_draws2 = _draw(policy, eval_ex, gen, pe["direct_samples"], rng, bs)
            s_dir_tp1 = _trait_matrix(policy, sae, layer, direct_draws2, all_ids, bs).mean(0)
            direct_drift = (s_dir_tp1 - s_dir_t)                            # (F,)

            omega_full = torch.exp(logp_tp1 - logp_t)                       # (N_price,)
            for N in pe["price_samples"]:
                w, s = omega_full[:N], s_price[:N]
                mean_w = w.mean()
                ws = (w[:, None] * s).mean(0)                               # (F,)
                price = ws - s.mean(0)                       # naive: assumes omega_bar = 1
                cov = ws - mean_w * s.mean(0)                # raw covariance (omega_bar-corrected)
                sn = (w[:, None] * s).sum(0) / w.sum() - s.mean(0)   # Hajek self-normalized
                ess = float((w.sum() ** 2) / (w ** 2).sum())
                for fi, fid in enumerate(feat_ids + ctrl_ids):
                    f.write(json.dumps({
                        "step": t, "feature_id": int(fid),
                        "is_control": fid in ctrl_ids, "N": N,
                        "direct_drift": round(float(direct_drift[fi]), 4),
                        "price": round(float(price[fi]), 4),
                        "cov": round(float(cov[fi]), 4),
                        "price_sn": round(float(sn[fi]), 4),
                        "mean_omega": round(float(mean_w), 4),
                        "omega_var": round(float(w.var()), 4),
                        "omega_max": round(float(w.max()), 4),
                        "ess": round(ess, 2),
                    }) + "\n")
            f.flush()   # persist each transition's rows so a time-kill can't lose them
            LOGGER.info(f"step {t}->{t+1} mean_omega={float(omega_full.mean()):.3f} "
                        f"ess@max={float((omega_full.sum()**2)/(omega_full**2).sum()):.1f}")
    LOGGER.info(f"done -> {jsonl_path}")

    from src.plotting import plot_from_jsonl
    plots_dir = out / f"plots{args.out_suffix}"
    plot_from_jsonl(jsonl_path, plots_dir)
    LOGGER.info(f"saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
