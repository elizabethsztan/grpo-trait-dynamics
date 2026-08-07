#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --job-name=pilot_price
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/slurm_logs/pilot_price_%j.out

# Phase 3 ONLY, on the checkpoints already produced by the earlier training jobs.
# Usage: sbatch lr_pilot_price.sh config_pilot_lr3e-5
set -eo pipefail
CFG=$1
if [ -z "$CFG" ]; then echo "usage: lr_pilot_price.sh <config_name>"; exit 2; fi
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment

echo "===== $CFG : PHASE 3 Price eval (re-run on existing checkpoints) ====="
uv run python -m experiments.run_price_eval --config experiments/configs/$CFG.yaml
echo "===== PRICE DONE: $CFG ====="
