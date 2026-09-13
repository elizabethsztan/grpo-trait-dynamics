# LLM binary confirmation preparation

This records the approved LLM arm of
`results/scratch/closure_synthesis_20260912/prospective_protocol.html`.
The calibration below has been performed once. No confirmation data have been
generated. This is a coefficient and specification freeze, pending review and
commit approval; the training/measurement launch path still requires the checks
listed below. Controlled confirmation is complete and unchanged. SAE stays gated.

## Development calibration

Only the three `hierarchical_above_hook_lr1e5_seed{290416,290417,290418}` runs under
`results/trajectory_forecast/hierarchical_20260724/prospective_v1/runs` are used.
[inputs.json](configs/llm_binary_confirmation/inputs.json) records their six source
file hashes, the derivation code hashes, and the calibration environment.
Its `source_base_commit` is the repository base for this preparation, not a claim
about the historical training source revision.

All three runs provide 60 unit-transition covariances but only 12 usable old-state
prevalences: t=0,5,...,55. The metrics record t supplies direct observed T_t;
record t+1 supplies C for t to t+1. The other 48 transitions per run have no
direct old-state T and are excluded without interpolation. Historical T and C
come from separate sampling panels (128 direct responses; 128 prompt groups
with two completions for the Price measurement). They are estimates, not exact
population quantities; the archives do not support prompt-group uncertainty for
these calibration coefficients.

[development_pairs.csv](configs/llm_binary_confirmation/development_pairs.csv)
contains the exact 36 input pairs and their source record numbers. Each candidate
is fitted once to these same pairs by minimizing

\[
\sum_t [C_t-T_t(1-T_t)\widehat S(T_t\text{ or }t)]^2.
\]

There is equal weight per covariance row, hence equal numbers of rows per run.
This is not least squares in raw S. The solver is NumPy 2.2.6
`linalg.lstsq(rcond=None)` with design `T*(1-T)*polyvander(predictor, degree)`.
No ratio normalization, trajectory fitting, model selection, or new objective is
introduced. The existing per-run development analyses precede this registered
pooled calibration; confirmation will show every run separately.

The full-precision source is
[coefficients.json](configs/llm_binary_confirmation/coefficients.json):

| Law | Frozen numerical coefficients |
| --- | --- |
| S=s | s=0.039956446653042224 |
| S(T)=a+bT | a=0.117667359804087; b=-0.16914768828119642 |
| S(t)=a+bt | a=0.07871491677261624; b=-0.001239007390641658 |

t is the source optimizer-update index, not a five-update measurement index.
These are development estimates; their scientific test is still prospective.

## Fixed confirmation specification

[confirmation.yaml](configs/llm_binary_confirmation/confirmation.yaml) retains
Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, LoRA strictly above hook 12,
learning rate 1e-5, and the archived model, reward, data, optimizer and generation
settings. Training rewards correct answers on the existing medium arithmetic
task with 90% correct hints. Evaluation uses wrong hints. The binary trait is
whether the parsed answer agrees with the hint; invalid answers count as zero.

Train exactly 60 updates for each seed 2026091501, 2026091502, 2026091503.
Their output paths are `results/llm_binary_confirmation/seed<seed>`; all three
were absent at calibration. There is no LLM initial-condition perturbation arm.
Measurement uses 512 independently sampled prompt groups, two completions each,
at every checkpoint 0 through 60, including the terminal pool. Retain exact
response tokens and prompt groups. Score each old-policy pool under the successor
policy: 61,440 successor scores per run. Preserve the existing binary likelihood
calculation and raw likelihood-ratio plug-in covariance; do not renormalize ratios.
Use separate measurement prompt and generation RNG streams, initialized at
training seed +200000 and +100000 respectively, isolated from training.

Each state law starts only from the measured T_0 and recursively applies
`T_next = T + T*(1-T)*S_hat`. The time rival additionally receives t=0,...,59.
No target-run fit, time shift or observed-state reset is permitted. Stop on a
nonfinite T or T outside [0,1], without clipping; retain the attempted invalid
state and failure count, and withhold full-horizon error for failed paths.

Compare all three models using per-run C and S closure residuals on transitions
0,...,59, and autonomous prevalence RMSE at checkpoints 5,10,...,60. Report raw S
only where T*(1-T)>0; boundary rows still support C diagnostics. Show trajectories,
terminal errors and completion for every run. Keep prompt-group joint bootstrap
measurement uncertainty separate from variation across the three training runs.
Good trajectories cannot rescue systematically wrong measured closures. These
replications compare the registered state/time laws but do not establish general
state-versus-schedule identification. No outcome-dependent exclusions or changes.

## Remaining implementation boundary

The YAML is a preparation specification, not a drop-in configuration for the
current `train_grpo_price.py`. Before launch, implement and independently check
the archived revision pin and output-exists protection, measurement RNG isolation,
dense grouped measurement including the terminal pool, token/group retention,
and analysis that consumes the frozen coefficients without fitting. Check the
registered likelihood and trait calculations on small unrelated fixtures. Freeze
bootstrap implementation details before any confirmation data exist. Do not use
reserved training seeds for implementation tests or launch from the legacy CLI.
These implementation checks must preserve the choices above; no LLM confirmation
launch is authorized by this preparation milestone.
