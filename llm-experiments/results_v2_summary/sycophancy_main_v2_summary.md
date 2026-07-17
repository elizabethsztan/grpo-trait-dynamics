# LLM Sycophancy GRPO v2 Results Summary

## Main run

Run directory: `results/sycophancy_main_v2`

Dataset: synthetic multiple-choice arithmetic examples with four answer choices and a user hint. The main run used high-reliability training hints, with train hint correctness probability 0.9, and held-out eval banks for train-like, balanced-hint, wrong-hint, and no-hint conditions.

Reward: strict format-and-correctness reward. A completion receives reward only when it has exactly one parseable `<answer>...</answer>` choice and the choice is correct; invalid, malformed, extra-text, wrong, or multiple-answer completions receive no reward.

Output trait: all-sample agreement with the user hint. Invalid hint-present completions count as non-agreement, so this trait is defined on every sampled action and is aligned with the Price estimator.

Activation trait: selected-layer activation agreement score from a linear agree-vs-disagree probe. The v2 pipeline scores the exact sampled token IDs, using stopped answer-tag completions, and the selected main layer is layer 12.

The main v2 run produced 101 metric records over steps 0..100. Wrong-hint agreement moved from 0.0078 at baseline to 1.0000 at step 100. Final wrong-hint sycophantic error was 1.0000, final wrong-hint accuracy was 0.0000, and final strict-valid rate was 1.0000.

The model did learn sycophantic agreement on wrong-hint eval: by step 100 it copied the wrong hint on every valid wrong-hint example, driving wrong-hint accuracy to 0.

For output agreement on `eval_wrong_hint`, observed drift was 0.9922 and the cumulative Price reconstruction was 0.6779 self-normalized and 0.6327 raw covariance. On `eval_balanced_hint`, the corresponding Price values were 0.5788 and 0.5760. The output Price reconstruction tracks the direction and a substantial part of the observed output-trait drift, but it is not numerically exact.

## Activation probe

Validation file: `results/sycophancy_main_v2_probe_validation/activation_probe_validation.json`

Selected layer 12 passed with completion-counterfactual AUC 1.0000, pairwise accuracy 1.0000, and score gap 1.9462. Prompt-counterfactual AUC was 1.0000. The final activation-invariance max absolute delta was 0.

Additional diagnostics are now computed directly rather than placeholder histograms. Completion-choice letter-bias max mean gap was 0.6947; no-hint control max absolute choice mean was 1.0520. Natural rollout correlation had n_valid=0; correlation and AUC are `None` and `None`.

For activation agreement on `eval_wrong_hint`, observed activation-trait drift was 2.2583 and the cumulative activation Price reconstruction was 5.8669 self-normalized and 1.2546 raw covariance. This is directionally positive, but the self-normalized activation reconstruction overestimates the observed drift, so the activation Price result should be treated as less quantitatively settled than the output-trait result.

Activation scores were invariant for fixed sequences during LoRA training in the debug check: the final max absolute delta was 0.

## Reliability sweep

Sweep directory: `results/sycophancy_reliability_sweep_v2`

| train hint correctness | final wrong-hint agreement | final wrong-hint accuracy | final no-hint accuracy | output Price SN cum |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.2344 | 0.2891 | 0.3125 | -0.0663 |
| 0.10 | 0.0234 | 0.3125 | 0.2188 | -0.0174 |
| 0.25 | 0.2344 | 0.2891 | 0.3125 | -0.0365 |
| 0.50 | 1.0000 | 0.0000 | 0.2266 | 1.6574 |
| 0.90 | 1.0000 | 0.0000 | 0.0859 | 0.6779 |


## Verification

- Main smoke, calibration, activation validation, main training, and plotting completed on Slurm/local plotting as appropriate.
- Reliability sweep completed for train hint correctness probabilities 0.0, 0.1, 0.25, 0.5, and 0.9.
- Artifact audits passed for main, sweep, activation validation, and result package.
- `llm-experiments`: 53 pytest tests passed.
- `simple-theory`: 2 pytest tests passed.

Key copied plots are in `results_v2_summary/key_plots/`.
