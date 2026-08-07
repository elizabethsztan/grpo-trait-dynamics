# LLM Sycophancy Trait Experiments

This experiment trains a small instruction-tuned LLM with outcome-only GRPO on synthetic multiple-choice arithmetic prompts where the user hint is usually correct. The reward is only final-answer correctness, while the measured unrewarded trait is agreement with the user hint. Counterfactual wrong-hint evaluations measure whether selected agreement becomes sycophantic error. A frozen activation agreement probe scores the same fixed prompt-completion behavior while LoRA is restricted to transformer layers above the probe hook.

## Install

```bash
uv sync
```

## Dry Calibration

This path validates data generation and writes calibration examples without loading a model.

```bash
cd llm-experiments
uv run python calibrate_base_model.py --config configs/qwen25_05b_sycophancy_smoke.yaml --dry-run
```

## Base-Model Calibration

```bash
uv run python calibrate_base_model.py --config configs/qwen25_05b_sycophancy_smoke.yaml
```

## Smoke Training

```bash
uv run python train_grpo_price.py --config configs/qwen25_05b_sycophancy_smoke.yaml
uv run python plot_results.py --run-dir results/sycophancy_smoke
```

## Main Training

```bash
uv run python train_grpo_price.py --config configs/qwen25_05b_sycophancy_main.yaml
uv run python plot_results.py --run-dir results/sycophancy_main
```

## Metrics And Plots

`metrics.jsonl` records train reward/accuracy, independent Price estimates, shuffled/null Price estimates, reused-rollout diagnostics, observed trait levels, and activation invariance diagnostics. Price-check plots compare observed trait drift against cumulative sampled `Cov(omega, s)` and show the residual. Wrong-hint plots track agreement with incorrect user hints, sycophantic error, and correct disagreement.

The activation probe is a fixed evaluator, not a steering intervention. It is built before training from balanced counterfactual prompt-completion pairs, and LoRA trainable parameters are asserted to live only in layers strictly above the hook.

## Safety Caveat

This is a benign synthetic sycophancy/deference experiment using arithmetic hints. It is not a harmful-content refusal, persuasion, or jailbreak experiment.
