#!/bin/bash
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=syco_main
#SBATCH --output=/cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-output-experiment/results/slurm_logs/%x_%j.out

# Run the sycophancy GRPO + Price experiment on one A100.
# Usage: sbatch slurm/main.sh [config-basename]   (default: qwen25_05b_sycophancy_main)
set -eo pipefail
CFG=${1:-qwen25_05b_sycophancy_main}

# Caches on the store (home cephfs quota is only 4 GiB); mirrors llm-activation-experiment.
export HF_HOME=/cephfs/store/gr-mc2473/eszt2/.hf-cache
export UV_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-cache
export UV_PYTHON_INSTALL_DIR=/cephfs/store/gr-mc2473/eszt2/.uv-python
export TRITON_CACHE_DIR=/cephfs/store/gr-mc2473/eszt2/.triton-cache
export XDG_CACHE_HOME=/cephfs/store/gr-mc2473/eszt2/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /cephfs/store/gr-mc2473/eszt2/trait-dynamics/grpo-trait-dynamics/llm-output-experiment

echo "===== $CFG : GRPO + Price ====="
# --no-sync: never re-resolve/rebuild the shared .venv at job start (concurrent
# `uv run` across nodes was rebuilding it mid-flight and breaking runs).
uv run --no-sync python train_grpo_price.py --config configs/$CFG.yaml
echo "===== TRAIN DONE: $CFG ====="

# Plot from the run dir (RunConfig.name -> results/<name>). Separate step because
# train_grpo_price.py only writes metrics.jsonl; plot_results.py renders the figures.
RUN_NAME=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('configs/$CFG.yaml'))['RunConfig']['name'])")
uv run --no-sync python plot_results.py --run-dir results/$RUN_NAME
echo "===== PLOTS DONE: results/$RUN_NAME/plots ====="
