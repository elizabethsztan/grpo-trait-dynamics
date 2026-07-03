import argparse, json, logging
from pathlib import Path
import numpy as np, torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores
from src.grpo import sample_completions
from src.verifier import reward, extract_answer
from src.data import load_gsm8k_splits, build_prompt

LOGGER = logging.getLogger(__name__)


def _gather(policy, sae, layer, examples, gen, K):
    S, R, lengths, parses = [], [], [], []
    for ex in examples:
        pid = policy.tokenizer(build_prompt(ex["question"], n_shots=gen["n_shots"]),
                               return_tensors="pt").input_ids[0]
        comps = sample_completions(policy, pid, K, gen)
        for c in comps:
            if c.numel() == 0:
                continue
            text = policy.tokenizer.decode(c, skip_special_tokens=True)
            S.append(collect_feature_scores(policy, sae, layer, pid, c).float().cpu().numpy())
            R.append(reward(text, ex["gold"]))
            lengths.append(len(c)); parses.append(1.0 if extract_answer(text) is not None else 0.0)
    return np.stack(S), np.array(R), np.array(lengths, float), np.array(parses)


def _corr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    fd, gen = cfg["FeatureDiscoveryConfig"], cfg["GenConfig"]
    policy = load_policy(cfg, "cuda")
    splits = load_gsm8k_splits(cfg)
    feat_ex = splits["feat"]
    cut = int(len(feat_ex) * fd["val_frac"])
    disc_ex, val_ex = feat_ex[:cut], feat_ex[cut:]

    best = None
    for layer in fd["candidate_layers"]:
        sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
        Sd, Rd, Ld, Pd = _gather(policy, sae, layer, disc_ex, gen, fd["K"])
        Sv, Rv, _, _ = _gather(policy, sae, layer, val_ex, gen, fd["K"])
        activity = (Sd > 0).mean(axis=0)
        rho_d = np.array([_corr(Sd[:, m], Rd) if activity[m] >= fd["min_activity"] else 0.0
                          for m in range(Sd.shape[1])])
        order = np.argsort(rho_d)
        neg_ids = [int(m) for m in order[:fd["top_k_each_sign"]]]
        pos_ids = [int(m) for m in order[::-1][:fd["top_k_each_sign"]]]

        def entry(m):
            return {"feature_id": m, "rho_discovery": round(float(rho_d[m]), 4),
                    "rho_val": round(_corr(Sv[:, m], Rv), 4), "activity": round(float(activity[m]), 4)}
        pos, neg = [entry(m) for m in pos_ids], [entry(m) for m in neg_ids]
        # keep only features whose sign holds on validation
        pos = [e for e in pos if e["rho_val"] > 0]; neg = [e for e in neg if e["rho_val"] < 0]
        strength = sum(abs(e["rho_val"]) for e in pos + neg)
        LOGGER.info(f"layer {layer}: kept {len(pos)}+/{len(neg)}- val-strength={strength:.3f}")

        rng = np.random.default_rng(cfg["ModelConfig"]["seed"])
        active_lowcorr = [m for m in range(Sd.shape[1])
                          if activity[m] >= fd["min_activity"] and abs(rho_d[m]) < 0.02]
        ctrl_ids = rng.choice(active_lowcorr, size=min(fd["n_controls"], len(active_lowcorr)),
                              replace=False).tolist() if active_lowcorr else []
        controls = [entry(int(m)) for m in ctrl_ids]
        nuisance = {"reward_vs_length": round(_corr(Rd, Ld), 4),
                    "reward_vs_parse": round(_corr(Rd, Pd), 4)}
        cand = {"layer_L": layer, "positive": pos, "negative": neg,
                "controls": controls, "nuisance": nuisance, "_strength": strength}
        if best is None or strength > best["_strength"]:
            best = cand

    best.pop("_strength")
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(best, open(out / "features.json", "w"), indent=2)
    LOGGER.info(f"chose layer_L={best['layer_L']} -> {out}/features.json")


if __name__ == "__main__":
    main()
