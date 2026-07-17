# Output-only LoRA coverage experiment

Run date: 2026-07-17

Branch: `adil/output-only-lora-coverage`

Implementation commits: `b96d9c7`, `0712483`

Historical code baseline: `582b4ce92d06793395f8bab9f52dd82e94791e20`

## Conclusions

1. The historical primary output-level result is exactly reproducible on the
   original L40S hardware. Observed trajectories (including no-hint), primary
   Price trajectories, completions, and the final adapter are identical. The
   shuffled-null Price diagnostic differs because disabling activation probing
   changes the shared diagnostic RNG stream; it is reported but is not treated
   as behavioral or primary Price evidence.
2. Changing the restricted-layer run from L40S to A100 preserves the wrong-hint
   and primary Price results within tolerance, but does not preserve no-hint
   generation. The A100 hardware control therefore fails the expanded overall
   gate.
3. On A100, applying LoRA to all 24 transformer layers preserves the final
   hinted behavior and primary Price result within tolerance. It substantially
   changes no-hint generation: final accuracy rises from `0.09375` to
   `0.1953125`, while invalid outputs fall from `0.6484375` to `0.21875`.
4. For this seed, the all-layer run first reaches wrong-hint agreement >= 0.9
   at step 10, versus step 25 for the restricted A100 run. The 15-step
   difference exceeds the predefined 10-step tolerance. This is a one-seed,
   five-step-resolution observation, and the parameter budget also increases
   from 2.02M to 4.40M, so it is not evidence for a general learning-rate claim.

The supported interpretation is narrower than "the results stay similar":
full-model LoRA reaches the same final hinted sycophancy and remains consistent
with the primary Price analysis, but it crosses the wrong-hint threshold earlier
in this seed and produces materially different, more valid no-hint outputs.

## Main runs

| Run | Slurm job | GPU | LoRA layers | Trainable params | Runtime | Result |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Historical reproduction | 208102 | NVIDIA L40S | 13-23 | 2,016,256 | 02:39:18 | Primary reproduction; PASS |
| Restricted A100 control | 208138 | NVIDIA A100-SXM4-40GB | 13-23 | 2,016,256 | 02:43:38 | Hinted PASS, no-hint FAIL |
| All-layer A100 | 208130 | NVIDIA A100-SXM4-40GB | 0-23 | 4,399,104 | 03:34:35 | Hinted endpoint PASS; timing/no-hint FAIL |

All jobs completed with exit code `0:0`, used model revision
`7ae557604adf67be50417f59c2c2f167def9a775`, ran for 100 training steps,
and produced 101 metric records plus a final adapter.

## Comparison results

The no-hint endpoint gates were added post-run during independent review after
the initial comparator was found to cover only `eval_wrong_hint`. They use the
same `0.05` endpoint-difference tolerance as the hinted accuracy/invalid gates
and should be interpreted as post-hoc completeness checks. The 10-step
wrong-hint crossing tolerance and primary Price tolerances were defined before
these main comparisons were evaluated.

| Metric | Historical reproduction | A100 hardware control | All layers vs restricted A100 |
| --- | ---: | ---: | ---: |
| Overall gate | PASS | FAIL | FAIL |
| Final agreement absolute difference | 0.000000 | 0.007812 | 0.000000 |
| Final accuracy absolute difference | 0.000000 | 0.000000 | 0.000000 |
| Final invalid-rate absolute difference | 0.000000 | 0.007812 | 0.000000 |
| Final no-hint accuracy absolute difference | 0.000000 | 0.062500 | 0.101562 |
| Final no-hint invalid-rate absolute difference | 0.000000 | 0.304688 | 0.429688 |
| No-hint accuracy trajectory RMSE | 0.000000 | 0.062033 | 0.091029 |
| No-hint invalid-rate trajectory RMSE | 0.000000 | 0.228071 | 0.322451 |
| Final observed-drift absolute difference | 0.000000 | 0.015625 | 0.000000 |
| Agreement trajectory RMSE | 0.000000 | 0.008693 | 0.097414 |
| Agreement >= 0.9 crossing-step difference | 0 | 5 | 15 |
| Final Price absolute difference | 0.000000 | 0.029344 | 0.129400 |
| Price trajectory RMSE | 0.000000 | 0.028464 | 0.130710 |
| Final shuffled-null Price absolute difference | 0.296234 | 0.099811 | 0.114934 |

For the direct same-A100 wrong-hint comparison, restricted and all-layer final
agreement are both `1.0`, final observed drift is `0.828125`, and final accuracy
and invalid-output rate are both `0.0`. Final cumulative primary Price is
`1.023160` for restricted LoRA and `0.893760` for all-layer LoRA. These values
do not summarize the separate no-hint distribution.

Detailed machine-readable comparisons and plots are stored alongside this
README:

- `historical_reproduction_comparison.{json,md}`
- `a100_hardware_control_comparison.{json,md}`
- `all_layers_vs_restricted_a100_comparison.{json,md}`
- `*_agreement_overlay.png` and `*_price_overlay.png`

## Smoke validation

| Run | Slurm job | GPU | Runtime | Records | Status |
| --- | ---: | --- | ---: | ---: | --- |
| Restricted layers | 208095 | NVIDIA L40S | 00:01:55 | 6 | COMPLETED 0:0 |
| All layers | 208100 | NVIDIA L40S | 00:00:57 | 6 | COMPLETED 0:0 |

The smoke summaries verified activation probing was disabled, the intended
layer indices were selected, and only the expected LoRA parameters were
trainable. A post-run safetensors key audit found all `77/77` expected
layer/module pairs in the restricted A100 adapter and all `168/168` in the
all-layer adapter, with no missing or extra pairs.

## Reproducibility and safety

The original result directory was never used as an output path. A SHA-256
manifest was captured before launching the runs and verified after completion:

```text
results_output_lora_summary/reference_manifest.json
verification result: PASS
```

The original ignored copy remains under
`results/sycophancy_output_lora_comparison_seed290402/`.

SHA-256 manifests for every complete candidate run are also tracked:

```text
historical_reproduction_run_manifest.json
a100_restricted_run_manifest.json
a100_all_layers_run_manifest.json
```

`run_metadata/` preserves each main run's resolved config, environment summary,
and Slurm output. Jobs 208095, 208100, 208102, and 208130 were submitted after
commit `b96d9c7` and before commit `0712483`; job 208138 was submitted 11 seconds
after `0712483`. The exact worktree state and interactive `srun` command lines
were not captured, so no stronger source-state claim is made. Future runs should
record both in the run summary. Slurm submit/start times, job IDs, partitions,
nodes, runtimes, and exit codes are recorded in `slurm_accounting.txt`;
application output is preserved in the copied logs.

Raw results remain in separate ignored directories:

```text
results/sycophancy_main_output_repro_seed290402
results/sycophancy_main_output_repro_a100_seed290402
results/sycophancy_main_output_all_layers_seed290402
results/sycophancy_output_lora_comparison_seed290402
```

## Verification

```text
llm-experiments: 33 passed
simple-theory:     2 passed
```

The two pytest suites were run in separate processes because both projects
currently expose a top-level package named `src`. The simple-theory project does
not declare pytest, so its tests were run explicitly with
`../llm-experiments/.venv/bin/pytest -q tests`.
