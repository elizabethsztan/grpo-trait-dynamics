#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=pilot_train
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/pilot_train_%j.out

# Phase 2 + Phase 3 for ONE lr config. Usage:
#   sbatch --dependency=afterok:<disc_jobid> lr_pilot_train.sh config_pilot_lr3e-5
# Requires features.json already present in that config's results dir (created
# by lr_pilot_discovery.sh).
set -e
CFG=$1
if [ -z "$CFG" ]; then echo "usage: lr_pilot_train.sh <config_name>"; exit 2; fi
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment

echo "===== $CFG : PHASE 2 GRPO ====="
uv run python -m experiments.run_grpo --config experiments/configs/$CFG.yaml
echo "===== $CFG : PHASE 3 Price eval ====="
uv run python -m experiments.run_price_eval --config experiments/configs/$CFG.yaml
echo "===== TRAIN DONE: $CFG ====="
