import argparse, json, logging
from pathlib import Path
import torch, yaml
from src.model import load_policy
from src.sae import load_sae
from src.trait import collect_feature_scores
from src.data import load_gsm8k_splits, build_prompt
from src.grpo import grpo_step

LOGGER = logging.getLogger(__name__)


def _assert_frozen(policy, sae, layer, pid, cid, before):
    after = collect_feature_scores(policy, sae, layer, pid, cid)
    drift = (after - before).abs().max().item()
    assert drift < 1e-4, f"TRAIT NOT FROZEN: max |Delta s|={drift} (LoRA leaked onto layer<=L?)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    grpo_cfg, gen = cfg["GRPOConfig"], cfg["GenConfig"]
    out = Path(cfg["OutputConfig"]["results_dir"]) / cfg["OutputConfig"]["name"]
    layer = json.load(open(out / "features.json"))["layer_L"]

    policy = load_policy(cfg, "cuda")
    policy.attach_lora(layer, grpo_cfg["lora_rank"])
    sae = load_sae(cfg["ModelConfig"]["sae_repo"], layer, "cuda")
    opt = torch.optim.Adam([p for p in policy.model.parameters() if p.requires_grad], lr=grpo_cfg["lr"])
    splits = load_gsm8k_splits(cfg)
    train = splits["train"]

    # fixed probe (prompt, completion) for the frozen-trait assertion
    probe_ex = train[0]
    probe_pid = policy.tokenizer(build_prompt(probe_ex["question"], n_shots=gen["n_shots"]),
                                 return_tensors="pt").input_ids[0]
    probe_cid = policy.tokenizer("Answer: 42", return_tensors="pt").input_ids[0]
    probe_before = collect_feature_scores(policy, sae, layer, probe_pid, probe_cid)

    metrics_path = out / "train_metrics.jsonl"
    rng = torch.Generator().manual_seed(grpo_cfg["seed"])
    with open(metrics_path, "w") as mf:
        policy.save_lora(out / "checkpoints" / "step_0000")
        for step in range(grpo_cfg["steps"]):
            idx = torch.randint(len(train), (grpo_cfg["batch_size"],), generator=rng).tolist()
            batch = [(train[i], policy.tokenizer(build_prompt(train[i]["question"], n_shots=gen["n_shots"]),
                     return_tensors="pt").input_ids[0]) for i in idx]
            m = grpo_step(policy, opt, batch, gen, cfg)
            _assert_frozen(policy, sae, layer, probe_pid, probe_cid, probe_before)
            m["step"] = step + 1
            mf.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}) + "\n")
            mf.flush()
            if (step + 1) % grpo_cfg["checkpoint_every"] == 0:
                policy.save_lora(out / "checkpoints" / f"step_{step + 1:04d}")
            LOGGER.info(f"step {step + 1}/{grpo_cfg['steps']} reward={m['mean_reward']:.3f} kl={m['kl']:.4f}")
    LOGGER.info(f"done -> {out}")


if __name__ == "__main__":
    main()
