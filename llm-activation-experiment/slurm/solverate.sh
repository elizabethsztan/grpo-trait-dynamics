#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=solverate
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/slurm_logs/solverate_%j.out

# Behavioural distribution-shift test: frozen base policy solve-rate on GSM8K vs SVAMP.
# Extra args pass through to run_solverate_compare:
#   sbatch solverate.sh config_real_lr1e-4 --n-prompts 128 --sources eval svamp
set -eo pipefail
CFG=${1:-config_real_lr1e-4}; shift || true
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
export UV_PYTHON_INSTALL_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-python
export TRITON_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.triton-cache
export XDG_CACHE_HOME=/cephfs/store/gr-mc2473/eszt2/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment

echo "===== $CFG : solve-rate compare (extra args: $*) ====="
uv run --no-sync python -m experiments.run_solverate_compare --config experiments/configs/$CFG.yaml "$@"
echo "===== SOLVERATE DONE: $CFG ====="
