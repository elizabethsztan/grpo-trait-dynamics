#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=real_train
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/slurm_logs/real_train_%j.out

# Phase 2 ONLY -- bank the checkpoint trajectory. Phase 3 is run separately
# (slurm/lr_pilot_price.sh) so its N budgets can be toggled without retraining.
# Requires results/<name>/features.json already present (reused from the pilot).
# Usage: sbatch real_train.sh config_real_lr1e-4
set -eo pipefail
CFG=${1:-config_real_lr1e-4}
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
# Redirect Triton's kernel-compile cache and any XDG cache off $HOME -- the home
# cephfs quota is only 4 GiB and Triton writes new kernels there by default, which
# blows the quota (Errno 122) mid-run. The store area is effectively unbounded.
export TRITON_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.triton-cache
export XDG_CACHE_HOME=/cephfs/store/gr-mc2473/eszt2/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment

echo "===== $CFG : PHASE 2 GRPO (train only) ====="
uv run python -m experiments.run_grpo --config experiments/configs/$CFG.yaml
echo "===== TRAIN DONE: $CFG ====="