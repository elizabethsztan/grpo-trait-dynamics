#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --job-name=lr_pilot
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/lr_pilot_%j.out

# lr pilot: sweep GRPO lr in {1e-5, 3e-5, 1e-4}, everything else fixed.
# Phase 0/1 are lr-independent -> run once and share features.json across runs.
set -e
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment
CFG=experiments/configs
BASE=config_pilot_lr1e-5      # its name (pilot_lr1e-5) hosts the shared Phase 0/1 outputs

echo "===== PHASE 0: capability (shared) ====="
uv run python -m experiments.run_capability_check --config $CFG/$BASE.yaml
echo "===== PHASE 1: feature discovery (shared) ====="
uv run python -m experiments.run_feature_discovery --config $CFG/$BASE.yaml

# share the discovered features with the other two lr runs
for name in pilot_lr3e-5 pilot_lr1e-4; do
  mkdir -p results/$name
  cp results/pilot_lr1e-5/features.json results/$name/features.json
done

for cfg in config_pilot_lr1e-5 config_pilot_lr3e-5 config_pilot_lr1e-4; do
  echo "===== $cfg : PHASE 2 GRPO ====="
  uv run python -m experiments.run_grpo --config $CFG/$cfg.yaml
  echo "===== $cfg : PHASE 3 Price eval ====="
  uv run python -m experiments.run_price_eval --config $CFG/$cfg.yaml
done
echo "===== LR PILOT DONE ====="
