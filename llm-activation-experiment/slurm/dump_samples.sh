#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=dump_samples
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/dump_samples_%j.out

# Dump a few real (prompt, action, reward) rollouts for the docs.
# Usage: sbatch dump_samples.sh config_pilot_lr1e-4
set -eo pipefail
CFG=${1:-config_pilot_lr1e-4}
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment

uv run python -m experiments.dump_samples --config experiments/configs/$CFG.yaml --n_prompts 2 --g 3
echo "===== DUMP DONE ====="