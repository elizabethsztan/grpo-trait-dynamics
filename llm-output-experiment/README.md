# LLM Sycophancy Trait Experiments

This experiment trains a small instruction-tuned LLM with outcome-only GRPO on synthetic multiple-choice arithmetic prompts where the user hint is usually correct. The reward is only final-answer correctness, while the measured unrewarded trait is agreement with the user hint. Counterfactual wrong-hint evaluations measure whether selected agreement becomes sycophantic error. A frozen activation agreement probe scores the same fixed prompt-completion behavior while LoRA is restricted to transformer layers above the probe hook.

## Install

Uses the shared repo-root uv environment — no per-experiment install. Run
everything with `uv run --no-sync` from inside this directory.

## Dry Calibration

This path validates data generation and writes calibration examples without loading a model.

```bash
cd llm-output-experiment
uv run --no-sync python calibrate_base_model.py --config configs/qwen25_05b_sycophancy_smoke.yaml --dry-run
```

## Base-Model Calibration

```bash
uv run --no-sync python calibrate_base_model.py --config configs/qwen25_05b_sycophancy_smoke.yaml
```

## Smoke Training

```bash
uv run --no-sync python train_grpo_price.py --config configs/qwen25_05b_sycophancy_smoke.yaml
uv run --no-sync python plot_results.py --run-dir results/sycophancy_smoke
```

## Main Training

```bash
uv run --no-sync python train_grpo_price.py --config configs/qwen25_05b_sycophancy_main.yaml
uv run --no-sync python plot_results.py --run-dir results/sycophancy_main
```

## Update-Rule Ablations (omega blow-up)

With LoRA on all layers, the first few AdamW steps move the policy so far that the Price
importance weights `omega = exp(post - pre)` blow up (mean 2-5.8, max 26-55 over steps 1-3),
which is what drives the predicted-vs-observed residual in the price-check plots. These three
10-step runs isolate the cause; each changes exactly one knob relative to
`configs/qwen25_05b_sycophancy_all_adilsampling_10step.yaml`.

- `lora13` — LoRA restricted to layers 13+ (`LoRAConfig.first_lora_layer: 13`), matching the
  collaborator's healthy run. Expect omega to recover and the residual to collapse.
- `allclip` — all layers, PPO-clipped ratio in 4 minibatches of 32
  (`TrainConfig.clip_range: 0.2`, `minibatch_size: 32`). Expect omega to *still* blow up: on the
  first minibatch of step 1 the ratio is identically 1, so clipping cannot bite.
- `allwarmup` — all layers, linear lr warmup over 5 steps (`TrainConfig.warmup_steps: 5`).
  A small first lr is the only lever that acts on step 1, so expect omega to be fixed.

```bash
cd llm-output-experiment
for c in lora13 allclip allwarmup; do sbatch --job-name=syco_$c slurm/main.sh qwen25_05b_sycophancy_${c}_10step; done
```

Verdict: read `results/<name>/metrics.jsonl` → `price.eval_balanced_hint.output_agreement`.
Healthy is steps 1-3 `mean_omega` ~ 0.9-1.5 with `max_omega` <~ 6; broken is `mean_omega` >~ 2,
`max_omega` >~ 20. For `allclip`, `train.clip_fraction` reports whether clipping ever engaged.

## Metrics And Plots

`metrics.jsonl` records train reward/accuracy, independent Price estimates, shuffled/null Price estimates, reused-rollout diagnostics, observed trait levels, and activation invariance diagnostics. Price-check plots compare observed trait drift against cumulative sampled `Cov(omega, s)` and show the residual. Wrong-hint plots track agreement with incorrect user hints, sycophantic error, and correct disagreement.

The activation probe is a fixed evaluator, not a steering intervention. It is built before training from balanced counterfactual prompt-completion pairs, and LoRA trainable parameters are asserted to live only in layers strictly above the hook.

## Safety Caveat

This is a benign synthetic sycophancy/deference experiment using arithmetic hints. It is not a harmful-content refusal, persuasion, or jailbreak experiment.
