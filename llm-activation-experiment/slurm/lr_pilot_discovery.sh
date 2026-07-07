#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=pilot_disc
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/slurm_logs/pilot_disc_%j.out

# Shared, lr-independent Phase 0 + Phase 1. Writes features.json under the
# 1e-5 run's dir and copies it to the other two so the training jobs can run
# in parallel off the same discovered features.
set -e
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment
CFG=experiments/configs

echo "===== PHASE 0: capability ====="
uv run python -m experiments.run_capability_check --config $CFG/config_pilot_lr1e-5.yaml
echo "===== PHASE 1: feature discovery ====="
uv run python -m experiments.run_feature_discovery --config $CFG/config_pilot_lr1e-5.yaml

for name in pilot_lr3e-5 pilot_lr1e-4; do
  mkdir -p results/pilots/$name
  cp results/pilots/pilot_lr1e-5/features.json results/pilots/$name/features.json
done
echo "===== DISCOVERY DONE (features shared to all 3 lr dirs) ====="
