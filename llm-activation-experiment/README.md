# LLM Activation Experiment (Part 3)

Tests the sampled Price-equation prediction for a frozen Qwen-Scope SAE trait
in Qwen3.5-2B-Base under GRPO on GSM8K: does `mean(omega*s) - mean(s)` recover
the direct trait drift `T_{t+1}-T_t`? See `experiment-spec.md`.

## Environment

Shared uv project at the repo root (`grpo-trait-dynamics/pyproject.toml`), one
`.venv` for both `simple-theory/` and this experiment. `flash-linear-attention`
is required — Qwen3.5's hybrid linear-attention layers use its Triton kernels
(the tiny `conv1d` step falls back to torch, which is negligible; `causal-conv1d`
is **not** needed and would require a mismatched CUDA build). Run on a GPU node.

## Pipeline (run in order)

```bash
cd llm-activation-experiment
# Phase 0 — capability gate (inspect success rate before continuing)
uv run python -m experiments.run_capability_check --config experiments/configs/config.yaml
# Phase 1 — pick frozen SAE trait features (+ controls) and layer L
uv run python -m experiments.run_feature_discovery --config experiments/configs/config.yaml
# Phase 2 — GRPO with LoRA strictly downstream of layer L (frozen trait)
uv run python -m experiments.run_grpo --config experiments/configs/config.yaml
# Phase 3 — Price-equation evaluation + plots
uv run python -m experiments.run_price_eval --config experiments/configs/config.yaml
```

On Swirles, wrap each command in `srun -p ampere --gres=gpu:1 --time=HH:MM:SS bash -lc '...'`.

Smoke test (tiny, one GPU, minutes): swap `config.yaml` → `config_smoke.yaml`.
Outputs under `results/<name>/`: `capability_report.json`, `features.json`,
`checkpoints/step_XXXX/`, `train_metrics.jsonl`, `price_eval.jsonl`, `plots/`.

## Key invariants
- **Frozen trait:** LoRA lives only on layers > L, so the layer-L SAE activation
  (hence the trait score s) is unchanged by training. Asserted every GRPO step.
- **Unbiased ω:** rollouts sampled at temperature 1.0 / top-p 1.0 / no top-k so the
  sampling distribution equals the logprob distribution entering ω. Price rollouts
  are drawn independently of the gradient rollouts.
