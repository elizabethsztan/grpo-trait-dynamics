# Full-LoRA Price calibration

## Outcome

No held-out final run was launched. All three n256 calibration candidates passed
the behavior gates but failed at least one Price-fit gate. The predeclared n512
fallback for the best behavior-passing candidate (learning rate `1e-5`) also
failed. Per the stopping rule, seed `290402` was not used for further tuning or
for a final run.

All reported fits compare real observed-evaluation checkpoints at steps
`0, 5, ..., 60` against Price predictions at those same steps. The historical
reference is `results/sycophancy_main`, pinned by
`results_output_lora_summary/reference_manifest.json`.

## Gates

Each distribution had to pass both historical-fit limits, where each limit is
the historical step-0-to-60 metric plus `0.02`:

| Distribution | RMSE limit | Final absolute residual limit |
| --- | ---: | ---: |
| Balanced hint | 0.050884 | 0.048864 |
| Wrong hint | 0.201354 | 0.193503 |

Both distributions also required final agreement at least `0.95` and final
invalid-output rate at most `0.05`.

## Calibration results

| Price samples | LR | Balanced RMSE | Balanced final residual | Wrong RMSE | Wrong final residual | Behavior | Overall |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 256 | 1e-5 | 0.055742 | 0.051603 | 0.141190 | 0.115221 | PASS | FAIL |
| 256 | 2e-5 | 0.069707 | 0.073783 | 0.168603 | 0.181202 | PASS | FAIL |
| 256 | 3e-5 | 0.165375 | 0.159177 | 0.224199 | 0.229801 | PASS | FAIL |
| 512 | 1e-5 | 0.156577 | 0.195321 | 0.109546 | 0.121082 | PASS | FAIL |

The n256 `1e-5` run was the deterministic fallback candidate, with selection
score `1.095476`. It narrowly missed the balanced-hint limits while passing both
wrong-hint limits. The n512 rerun produced a worse balanced-hint trajectory and
did not improve normalized importance diagnostics overall. Minimum ESS
fractions were `0.5583` balanced and `0.7000` wrong, while maximum importance
weights were `20.0855` and `12.1825`, respectively. Compared with the n256
`1e-5` run, only wrong-hint minimum ESS fraction improved; balanced minimum ESS
fraction, both median ESS fractions, both mean-weight deviations, and the
balanced maximum weight worsened; the wrong-hint maximum was unchanged.
Absolute ESS increased because the sample count doubled.

Increasing Price sampling changes how many stochastic generations occur before
each training rollout in the current implementation. The n512 run is therefore
an independent stochastic training trajectory, not merely a lower-variance
re-estimate of the n256 trajectory. Its failure is still valid for the planned
fallback gate, but should not be interpreted as a paired sampling-only
comparison.

## CUDA allocator incident

The initial two-step smoke used a reduced `4 x 4` training batch and passed as
job `209895`. The first three full-batch calibration attempts then reproduced an
identical CUDA OOM during the third backward pass on two different A100 nodes:

```text
Tried to allocate: 3.26 GiB
PyTorch reserved but unallocated: 6.99 GiB
```

Jobs `209897`, `209898`, and `209899` were preserved as incomplete failed runs.
A four-step `16 x 8` memory smoke, job `210031`, crossed the failure point and
completed after the operator launched it with `expandable_segments:True`. That
launch environment was not captured inside the smoke artifacts, so this is
operator-recorded rather than artifact-proven provenance. The subsequent
committed training entry point applies the exact allocator policy from config
before importing PyTorch, rejects inherited alternatives, records it in
`summary.json`, and validates it during scoring.

## Runs and operator-recorded source states

| Purpose | Job | Source commit | Result directory |
| --- | ---: | --- | --- |
| Reduced-batch smoke | 209895 | `5632d00181e9ffc8536a95ba6029d037937b2a81` | `results/sycophancy_full_lora_price_smoke_lr2e5_n256_seed290403` |
| Initial 1e-5 attempt | 209897 | `5632d00181e9ffc8536a95ba6029d037937b2a81` | `results/sycophancy_full_lora_price_cal_lr1e5_n256_seed290403` |
| Initial 2e-5 attempt | 209898 | `5632d00181e9ffc8536a95ba6029d037937b2a81` | `results/sycophancy_full_lora_price_cal_lr2e5_n256_seed290403` |
| Initial 3e-5 attempt | 209899 | `5632d00181e9ffc8536a95ba6029d037937b2a81` | `results/sycophancy_full_lora_price_cal_lr3e5_n256_seed290403` |
| Full-batch memory smoke | 210031 | `2f235dde987aa9b524dd040c8c96c780c4c26d55` | `results/sycophancy_full_lora_price_memory_smoke_lr2e5_n256_seed290403` |
| n256 1e-5 calibration | 210104 | `9f3fcfc900b3e3392b22b0688cd5c8a862ecd22f` | `results/sycophancy_full_lora_price_cal_lr1e5_n256_seed290403_retry1` |
| n256 2e-5 calibration | 210105 | `9f3fcfc900b3e3392b22b0688cd5c8a862ecd22f` | `results/sycophancy_full_lora_price_cal_lr2e5_n256_seed290403_retry1` |
| n256 3e-5 calibration | 210106 | `9f3fcfc900b3e3392b22b0688cd5c8a862ecd22f` | `results/sycophancy_full_lora_price_cal_lr3e5_n256_seed290403_retry1` |
| n512 1e-5 fallback | 210412 | `ce40d961ec5a4135833879257dbb5306ab6c4ca5` | `results/sycophancy_full_lora_price_cal_lr1e5_n512_seed290403_retry1` |

The source commits were recorded by the operator from commit-before-submit
timing and matching canonical configs. The run artifacts do not embed Git SHA,
worktree status, or exact launch commands, so the table is not an independent
artifact-level proof of exact source state.

The n256 jobs each produced exactly 61 metric records and used n256 at every
Price step. The fallback produced exactly 61 records and used n512 at every
Price step. Every completed calibration summary reports all 24 LoRA layers and
`cuda_allocator_config: expandable_segments:True`.

## Artifacts

- `price_fit/` contains machine-readable reports, gate summaries, checkpoint-only plots, and the n256 selection record.
- `configs/` contains every canonical smoke, initial, retry, and fallback config.
- `run_metadata/` contains application logs and Slurm accounting.
- `manifests/` contains SHA-256 manifests for every raw output directory, including incomplete OOM attempts.

All nine run manifests and the original historical-reference manifest were
verified after the experiments. Raw outputs remain in ignored, non-overlapping
`results/` directories.

## Verification

```text
llm-experiments: 47 passed
simple-theory:     2 passed
independent review: no blocking or important findings before launch
```

The test suites are run separately because both projects expose a top-level
package named `src`.
