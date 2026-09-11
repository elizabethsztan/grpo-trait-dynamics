import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores_batch
from src.logprob import generation_logprob_batch
from src.grpo import sample_completions_batch
from src.data import load_prompts, build_prompt

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
        out.extend(generation_logprob_batch(policy, chunk))
    return torch.tensor(out, dtype=torch.float64)


def _dump_pool(policy, sae, layer, eval_ex, gen, ckpts, n_trans, N, bs, rng,
               feat_ids, ctrl_ids, all_ids, out, suffix):
    # Raw per-rollout pool for offline bootstrap bands. Per transition we draw N rollouts
    # from pi_t ONCE and save omega = pi_{t+1}/pi_t and trait scores s; smaller-n bands are
    # then obtained offline by resampling n-with-replacement from this pool, and the full-N
    # estimate serves as the low-variance reference line. Direct estimator skipped by design.
    omega_all, s_all, steps = [], [], []
    for t in range(n_trans):
        step_t, step_tp1 = (int(ckpts[t].name.split("_")[1]),
                            int(ckpts[t + 1].name.split("_")[1]))
        policy.load_lora(ckpts[t])
        draws = _draw(policy, eval_ex, gen, N, rng, bs)
        s = _trait_matrix(policy, sae, layer, draws, all_ids, bs)           # (N, F)
        logp_t = _logprobs(policy, draws, bs)
        policy.load_lora(ckpts[t + 1])
        logp_tp1 = _logprobs(policy, draws, bs)
        omega = torch.exp(logp_tp1 - logp_t)                               # (N,)
        omega_all.append(omega.numpy()); s_all.append(s.numpy()); steps.append(step_t)
        ess = float((omega.sum() ** 2) / (omega ** 2).sum())
        LOGGER.info(f"step {step_t}->{step_tp1} pool N={N} mean_omega={float(omega.mean()):.3f} ess={ess:.1f}")
    path = out / f"pool{suffix}.npz"
    ids = feat_ids + ctrl_ids
    # ckpt_steps = the n_trans+1 checkpoint step numbers spanned (for a real-step x-axis).
    ckpt_steps = [int(ckpts[t].name.split("_")[1]) for t in range(n_trans + 1)]
    np.savez(path,
             omega=np.stack(omega_all),                                    # (T, N)
             s=np.stack(s_all),                                            # (T, N, F)
             feat_ids=np.array(ids),
             is_control=np.array([f in ctrl_ids for f in ids]),
             steps=np.array(steps), ckpt_steps=np.array(ckpt_steps), layer=layer)
    LOGGER.info(f"dumped pool -> {path}  (T={len(steps)}, N={N}, F={len(ids)})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    # --out-suffix: write price_eval<suffix>.jsonl / plots<suffix>/ so repeated Phase-3
    # sweeps on the same checkpoints don't clobber each other.
    # --max-transitions: cap the number of adjacent checkpoint pairs evaluated (0 = all);
    # set to 1 for a cheap single-pair N-convergence diagnostic.
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-transitions", type=int, default=0)
    # --dump-pool: per transition, draw max(price_samples) rollouts from pi_t once and save
    # the raw per-rollout omega and trait scores s to pool<suffix>.npz -- for offline
    # bootstrap error bands (resample n from the pool). Skips the direct estimator entirely
    # (bands come from the pool; the full-N estimate is the low-variance reference line).
    ap.add_argument("--dump-pool", action="store_true")
    # --prompt-split: which split the price/direct rollouts are drawn from. Default 'eval'
    # (held-out test) reproduces the headline. 'train' draws from the trained-on data, so
    # cov estimates the trait drift on the TRAIN distribution -- overlay it on the eval trait
    # curve to test whether cov from training-distribution rollouts predicts held-out drift.
    ap.add_argument("--prompt-split", default="eval", choices=["eval", "train", "feat", "svamp"])
    # --step-stride: evaluate every K-th checkpoint instead of every one, so each transition
    # spans K GRPO steps (omega = pi_{t+K}/pi_t). Fewer, coarser transitions -> cheaper and
    # fewer plot points; the Price identity still holds (a bigger jump just makes omega more
    # heavy-tailed / lowers ESS). The final checkpoint is always kept so the trait curve
    # reaches the trained endpoint (the last interval may then be shorter than K).
    ap.add_argument("--step-stride", type=int, default=1)
    # --transmission: also recompute the trait s on the SAME price draws under pi_{t+1}
    # and log the Price transmission term E[omega*(s_{t+1}-s_t)] per feature. Zero by
    # construction for frozen (lora_layers=">L") runs; nonzero when LoRA trains through
    # layer L. Full decomposition then reads ΔT ≈ cov (selection) + transmission. Costs
    # one extra SAE-scoring pass per transition (no extra generation).
    ap.add_argument("--transmission", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    pe, gen = cfg["PriceEvalConfig"], cfg["GenConfig"]
    if gen["temperature"] != 1 or gen["top_p"] != 1 or gen["top_k"] not in (0, None):
        raise ValueError("Price evaluation requires temperature=1, top_p=1, top_k=0.")
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    features = json.load(open(out / "features.json"))
    layer = features["layer_L"]
    feat_ids, ctrl_ids = _feature_ids(features)
    all_ids = torch.tensor(feat_ids + ctrl_ids)

    policy = load_policy(cfg, "cuda")
    # No attach_lora here: the first load_lora(ckpt) wraps the base from the saved
    # adapter config, and subsequent load_lora calls swap adapter weights in place.
    sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
    eval_ex = load_prompts(cfg, args.prompt_split)[: pe["eval_prompts"]]
    LOGGER.info(f"drawing price/direct rollouts from split='{args.prompt_split}' ({len(eval_ex)} prompts)")

    all_ckpts = sorted((out / "checkpoints").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    stride = max(1, args.step_stride)
    ckpts = all_ckpts[::stride]
    if ckpts[-1] != all_ckpts[-1]:          # always reach the trained endpoint
        ckpts.append(all_ckpts[-1])
    if stride > 1:
        LOGGER.info(f"step-stride={stride}: {len(ckpts) - 1} transitions over checkpoint steps "
                    f"{[int(c.name.split('_')[1]) for c in ckpts]}")
    rng = np.random.default_rng(cfg["ModelConfig"]["seed"] + 777)
    # Price sweep only consumes up to max(price_samples) rollouts; the direct estimate
    # uses its own separate direct_samples draws. (Don't over-draw to direct_samples.)
    N_price = max(pe["price_samples"])
    bs = pe.get("batch_size", 32)

    n_trans = len(ckpts) - 1
    if args.max_transitions > 0:
        n_trans = min(n_trans, args.max_transitions)

    if args.dump_pool:
        _dump_pool(policy, sae, layer, eval_ex, gen, ckpts, n_trans, N_price, bs, rng,
                   feat_ids, ctrl_ids, all_ids, out, args.out_suffix)
        return

    jsonl_path = out / f"price_eval{args.out_suffix}.jsonl"
    with open(jsonl_path, "w") as f:
        for t in range(n_trans):
            step_t, step_tp1 = (int(ckpts[t].name.split("_")[1]),
                                int(ckpts[t + 1].name.split("_")[1]))
            policy.load_lora(ckpts[t])
            # price draws from pi_t (independent of the gradient rollouts) + trait scores.
            # Preserve generation batches when scoring both checkpoints. BF16 full-pass
            # errors need not cancel in omega, so use the cached generation path.
            price_draws = _draw(policy, eval_ex, gen, N_price, rng, bs)
            s_price = _trait_matrix(policy, sae, layer, price_draws, all_ids, bs)   # (N_price, F)
            logp_t = _logprobs(policy, price_draws, bs)
            # direct T_t (high budget)
            direct_draws = _draw(policy, eval_ex, gen, pe["direct_samples"], rng, bs)
            s_dir_t = _trait_matrix(policy, sae, layer, direct_draws, all_ids, bs).mean(0)   # (F,)

            policy.load_lora(ckpts[t + 1])
            logp_tp1 = _logprobs(policy, price_draws, bs)
            # Re-score the SAME price draws under pi_{t+1}: byte-identical completions, so
            # ds = s_{t+1}(a) - s_t(a) isolates the trait change from the weights <=L moving.
            s_price_tp1 = (_trait_matrix(policy, sae, layer, price_draws, all_ids, bs)
                           if args.transmission else None)
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
                # transmission = E[omega*(s_{t+1}-s_t)] on the same draws; 0 for frozen runs.
                trans = ((w[:, None] * (s_price_tp1[:N] - s)).mean(0)
                         if s_price_tp1 is not None else None)
                for fi, fid in enumerate(feat_ids + ctrl_ids):
                    row = {
                        "step": step_t, "step_end": step_tp1, "feature_id": int(fid),
                        "is_control": fid in ctrl_ids, "N": N,
                        "direct_drift": round(float(direct_drift[fi]), 4),
                        "price": round(float(price[fi]), 4),
                        "cov": round(float(cov[fi]), 4),
                        "price_sn": round(float(sn[fi]), 4),
                        "mean_omega": round(float(mean_w), 4),
                        "omega_var": round(float(w.var()), 4),
                        "omega_max": round(float(w.max()), 4),
                        "ess": round(ess, 2),
                    }
                    if trans is not None:
                        row["transmission"] = round(float(trans[fi]), 4)
                    f.write(json.dumps(row) + "\n")
            f.flush()   # persist each transition's rows so a time-kill can't lose them
            LOGGER.info(f"step {step_t}->{step_tp1} mean_omega={float(omega_full.mean()):.3f} "
                        f"ess@max={float((omega_full.sum()**2)/(omega_full**2).sum()):.1f}")
    LOGGER.info(f"done -> {jsonl_path}")

    from src.plotting import plot_from_jsonl
    plots_dir = out / f"plots{args.out_suffix}"
    plot_from_jsonl(jsonl_path, plots_dir)
    LOGGER.info(f"saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
