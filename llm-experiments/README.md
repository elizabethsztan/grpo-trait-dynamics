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

## Sampling and likelihood accounting

Sequence likelihoods use float32 log-softmax and float32 sequence sums even when
model logits are bfloat16. For samples from `generate_completions`, scoring uses
the recorded EOS ID and minimum generation length: EOS is suppressed and the
remaining tokens are renormalized during the first `min_new_tokens` positions.
The stopping EOS is included in the likelihood, including when PAD and EOS share
an ID. A pad token sampled before termination remains part of the response.

Generation explicitly supports sampling with temperature 1, top-p 1, top-k 0,
and repetition penalty 1. Other sampling transformations and unknown generation
settings are rejected because this scorer does not implement their likelihoods.
Model-specific generation presets are not inherited. The model's generation
configuration is restored after each generation call, including on failure.
The prompt format, answer parser, reward, and empirical covariance are unchanged.

Token-only samples without generation metadata retain raw-policy scoring for
compatibility. They must not be substituted for generated samples in an
accounting run that suppresses EOS. Historical likelihoods and trajectories are
not corrected retrospectively by this change.

## Paper study collection pipeline

`run_paper_study.py` separates training from measurement. The development
configuration is `configs/qwen25_05b_sycophancy_paper.yaml`; its final measurement
budget and paper execution flag remain subject to pilot review. The historical
runner rejects this configuration so its older output behavior cannot bypass
the study manifest and checkpoint collection.

The following commands describe the pilot workflow after code review. Preparing
a study creates data and run entries only; it does not load a model or train:

```bash
python run_paper_study.py prepare \
  --config configs/qwen25_05b_sycophancy_paper.yaml \
  --cohort pilot --output results/paper_pilots_v1
python run_paper_study.py train --study results/paper_pilots_v1 \
  --run-id hint_0.25_seed2026092291
python run_paper_study.py measure --study results/paper_pilots_v1 \
  --run-id hint_0.25_seed2026092291 --measurement-id n512
python run_paper_study.py status --study results/paper_pilots_v1
```

The second pilot run ID is `hint_0.90_seed2026092292`. Model and tokenizer loading
uses only locally cached files at the pinned revision. Training applies all-layer
LoRA with the existing simplified objective, records every training response,
and saves initial and per-update adapters. No measurement occurs during training.
The no-hint condition uses the original no-hint prompt and correctness reward.

`--cohort paper` prepares thirty entries (five seeds each at 10%, 25%, 50%, 75%,
90%, and no-hint training). Training those entries is refused while
`paper_execution_enabled` is false. Enable it only after the agreed pilot review
and final configuration approval, in a newly prepared study. Do not edit the
manifest of an existing trained study: training artifacts bind to its hash.

The shared bank contains 2,048 matched problems, with wrong-hint, correct-hint,
balanced-hint, and no-hint variants. Its even-sized prefixes remain exactly
balanced in hint correctness for the balanced variant. The default collection
uses the first 512 groups and two responses per group. After discussing pilot
precision, a larger measurement can reuse the same saved policies:

```bash
python run_paper_study.py measure --study results/paper_pilots_v1 \
  --run-id hint_0.25_seed2026092291 --measurement-id n1024 --prompts 1024
```

Changing the measurement ID alone performs a deterministic replay, not an
independent replication. Increasing the prompt count retains the original sample
prefix. Separate random streams distinguish direct evaluations from Price pools,
and measurement restores the caller's Python, NumPy, and Torch random states.
Source and successor likelihoods use the same prompt-group batch boundaries.
Measurement refuses a changed weight dtype or generation configuration.

Each attempt has a `status.json` recording completion or failure, timing, source
hashes, commit, and package versions. Existing attempt directories are never
overwritten or resumed. A process killed without cleanup may retain `running`
status; that is not a completed attempt. Source pools are written before successor
scoring so a failed transition retains the collected responses. Checkpoint and
bank hashes are verified before replay.

Raw `observed_*.jsonl.gz` and `source_*.jsonl.gz` files include full prompts,
response tokens/text, traits, stopping metadata, group/response indices, source
checkpoint identity, and old likelihoods. Paired `price_*.jsonl.gz` files add the
successor likelihood, successor checkpoint identity, and source-pool hash.
`files.json` inventories the completed measurement files and their checksums.
`config.json` and `runtime.json` record requested and resolved settings.

`metrics.jsonl` uses checkpoint numbers: step zero has no Price increment; step
one contains the 0-to-1 increment. Observed evaluations occur at 0, 5, ..., 100
under the default configuration; missing evaluations remain absent. Always
compare cumulative covariance and observed drift at the same checkpoint.
Bootstrap uncertainty, reporting, and laptop export are a subsequent milestone;
the collection command does not perform fitted modeling or launch other runs.

## Safety Caveat

This is a benign synthetic sycophancy/deference experiment using arithmetic hints. It is not a harmful-content refusal, persuasion, or jailbreak experiment.
