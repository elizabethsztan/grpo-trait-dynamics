# Price closure study: first diagnostic milestone

Build a small dynamical model of unrewarded trait change by starting from the
Price equation, identifying what remains unclosed, and using experiments to test
closure assumptions and estimate their constants.

Both binary and continuous traits are central. Seek reusable closure forms with
setting-specific constants. Evaluate the resulting models through their
interpretation, measured closure residuals, and generated trajectories.
Incremental forecasting gains from Price measurements are a secondary question.

This implementation stops at **per-run measurements and diagnostic plots**.
It does not fit or select closures, pool coefficients, start training, or load LLMs.
Discuss the diagnostics before choosing the next phase. Estimate relationships
per run first; shared setting-level coefficients require evidence of consistency.

## Running

From the repository root, use a CPU-only environment; a training environment is
not needed:

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r closure_study/requirements.txt
.venv/bin/python -m pytest closure_study/tests -q
.venv/bin/python -m closure_study --output results/closure_study/diagnostics_v1
```

An existing nonempty output directory is never overwritten. Use a new directory
for a rerun. The command generates the inventory, diagnostics, and report together;
after changing the analysis code, rerun it in a new output directory.

`--registry` selects a YAML/JSON registry (default `configs/existing.yaml`).
Registry paths resolve relative to its `root`, which itself resolves relative to
the registry file. The default explicitly selects historical development data
used by the prior report; no wildcard searches open untouched runs. It includes
one externally stored SAE run. Missing sources are listed in the inventory and
report. Malformed input can stop the run; inspect the offending archive before
rerunning. Sources rejected during diagnostic extraction are listed with reasons.

## Sources and implementation reuse

The archive contracts were inspected in `trajectory_forecast/artifacts.py` and
`simple_artifacts.py` on `adil/hierarchical-trait-forecast` (`30df04c`), and in the
saved source worktrees. The new adapters deliberately preserve missing levels
and arbitrary checkpoint spacing. Historical loaders that accumulate Price
increments into observed levels or require unit intervals are not reused.

The default registry covers the four systems: tabular binary, neural continuous,
LLM binary wrong-hint agreement, and LLM continuous SAE maxima. Different SAE
features from a model run remain one run, not independent training replicates.
The recorded code/configuration/seed metadata do not establish that global
generation or initialization randomness was independently seeded. Reference
source hashes are not assertions of the actual training revision.

## Definitions and interpretation

For old-population weights `p`, fixed trait `z`, and the **raw** likelihood ratio
`w`, the exports use:

```
mu = E_p[z]; V = E_p[(z-mu)^2]; M3 = E_p[(z-mu)^3]
C = Cov_p(w,z); Q = Cov_p(w,(z-mu)^2)
beta = C/V; S = C/[T(1-T)] for binary traits
Delta mu = C; Delta V = Q-C^2                 [population identities]
Q approximately beta*M3                       [additional linear-reweighting assumption]
```

`Q_minus_beta_M3` is a diagnostic of that additional assumption; computing
`Delta mu = beta*V` with beta defined as C/V cannot validate it. Conditional
plots compare raw binned mean ratios with `1 + beta*(z-mu)` at fixed
early/middle/late transitions. They use population-weighted quantile bins;
normalization error can also contribute to their residuals.

Ratios are never normalized to average one for the primary C/Q calculation.
`C_known_normalizer` and the biased/nonlinear `C_self_normalized` are separately
named diagnostics. The primary sampled estimates are plug-in covariances, not
unbiased finite-sample covariances. Empirical ratio ESS is descriptive and is not
a confidence interval or a guarantee of tail coverage.

All calculations and numeric exports use float64, but historical float32 source
precision cannot be recovered. Plot labels round for display only.

### Source-specific limitations

- **Tabular:** all binary trait moments are determined by T. Q is algebraic,
  so a binary Q/moment equality is not an independent linearity test.
- **Neural:** frozen scores and pre-transition weights recover exact finite-table
  moments. Adjacent saved weight vectors recover moment fluxes. The final
  transition has no saved next-state weights, so its archived mean increment is
  retained while Q and next variance remain missing. Storage support failures
  are marked, and ratios are not plotted when absolute continuity fails.
- **Output LLM:** historical rows at step j describe the update j-1 to j. Direct
  prevalence is sparse; S is shown only when an old-state observation exists.
  No interpolation, carried-forward state, or accumulated-covariance level is
  substituted. Invalid parsed answers count as non-agreement, following the
  original binary trait definition.
- **Output accounting:** unit covariances are summed between direct observation
  checkpoints. This telescoping accounting check does not create an aggregate
  reweighting coefficient S for that larger interval.
- **SAE:** raw old-policy pools recover C, V, M3, and Q; the last checkpoint has
  no corresponding old-policy pool. Separate direct-mean archives, where present,
  support an independent mean-change check. Remaining historical direct drifts
  are rounded, with weaker panel provenance. New-score and direct-variance samples
  are unavailable. Above-hook score invariance is designed but not rechecked here.
  One legacy pool lacks endpoints. Its explicit endpoint supplement is bound to
  the pool SHA-256 and checked against its saved checkpoint bank; no generic
  missing-endpoint fallback assumes unit spacing.

Sampled sources lack the prompt/panel records needed for robust uncertainty
propagation; this milestone does **not** invent confidence bands. Numeric zero
variance yields missing coefficients. Nonzero coefficients can still be poorly
resolved near a boundary; their precision is explicitly unassessed.

## Output schema (version 1)

- `inventory.json`, `run_inventory.csv`: one entry per recorded run, paths and
  hashes, configuration, model/decoding metadata, recorded seeds, and limitations.
  Tabular update size `eta` is recorded separately from optimizer `learning_rate`.
- `transitions.jsonl`, `transitions.csv`: one row per run/trait/transition,
  actual `step`, `step_end`, `interval`, state/moment/flux fields, measurement
  source, uncertainty status, and missingness. JSON missing fields/nulls and CSV
  empty cells are missing, never implied zero. `population_weights` and `omega`
  have different meanings; the registry adapters keep them separate.
- `conditional_bins.*`: bin means, probability mass, raw mean ratio, linear
  reference value and residual. Sample counts are not training replicate counts.
- `accounting.*`: independently observed or enumerated changes versus matched
  covariance sums, with the comparison type recorded.
- `source_issues.*`: missing/malformed artifacts and reasons for exclusion.
- `manifest.json`: registry/source-code hashes and analysis environment;
  `artifact_manifest.json`: hashes of generated results.
- `accounting_audit.md`, `report.md`, `index.html`, `plots/`: reviewable report and
  PNG/PDF figures. Per-run plots preserve identities; overlays separate settings,
  feature definitions and interval lengths. No coefficient estimates are pooled.

## Review and commit agreement

Before every commit, run the relevant checks and obtain a separate independent
review of the exact proposed diff and scientific assumptions. The reviewer must
not edit files. If the reviewer raises any issue, pause for user discussion
**before making fixes**. After agreed corrections, recheck and review the revised
diff. Even after a clean review, pause with the changes, test results, review
outcome, and a detailed proposed commit title/body. Commit only after explicit
approval. The diagnostic milestone also ends with a discussion before fitting,
pooling coefficients, or adding experiments.
