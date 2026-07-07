#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=real_price
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment/results/slurm_logs/real_price_%j.out

# Phase 3 ONLY on the banked real_lr1e-4 checkpoints. Full-trajectory light pass
# (~6.4h). Optional args are passed through to run_price_eval:
#   sbatch real_price.sh config_real_lr1e-4                       # full 20 transitions
#   sbatch real_price.sh config_real_lr1e-4 --out-suffix _s1      # a re-sample
#   sbatch real_price.sh config_real_lr1e-4 --max-transitions 1 --out-suffix _conv
set -eo pipefail
CFG=${1:-config_real_lr1e-4}; shift || true
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
# Keep the uv-managed Python on the store too (home is 4 GiB): the .venv is built
# against this standalone 3.13 (bundles its own headers, so Triton compiles on the
# header-less compute nodes -- unlike system /usr/bin/python3.10).
export UV_PYTHON_INSTALL_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-python
# Redirect Triton's kernel-compile cache and any XDG cache off $HOME -- the home
# cephfs quota is only 4 GiB and Triton writes new kernels there by default, which
# blows the quota (Errno 122) mid-run. The store area is effectively unbounded.
export TRITON_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.triton-cache
export XDG_CACHE_HOME=/cephfs/store/gr-mc2473/eszt2/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-activation-experiment

echo "===== $CFG : PHASE 3 Price eval  (extra args: $*) ====="
# --no-sync: never re-resolve/rebuild the shared .venv at job start (see real_train.sh).
uv run --no-sync python -m experiments.run_price_eval --config experiments/configs/$CFG.yaml "$@"
echo "===== PRICE DONE: $CFG ====="