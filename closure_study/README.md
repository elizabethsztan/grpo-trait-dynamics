# Price closure study

Build a small dynamical model of unrewarded trait change by starting from the
Price equation, identifying what remains unclosed, and using experiments to test
closure assumptions and estimate their constants.

Both binary and continuous traits are central. Seek reusable closure forms with
setting-specific constants. Evaluate the resulting models through their
interpretation, measured closure residuals, and generated trajectories.
Incremental forecasting gains from Price measurements are a secondary question.

The diagnostic command stops at **per-run measurements and diagnostic plots**.
A separate, small fitting command implements the subsequently agreed comparison
on the two controlled systems. Neither command pools coefficients, starts training,
or loads LLMs. Shared setting-level coefficients require evidence of consistency.

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

## Controlled closure comparison

After discussing the diagnostics, fit the approved candidates to each controlled
run independently:

```bash
.venv/bin/python -m closure_study.fit_controlled \
  --table results/closure_study/diagnostics_v3/transitions.jsonl \
  --output results/closure_study/controlled_fits_v1
```

- Binary: constant S, affine/quadratic S(T), and affine/quadratic S(t). Each
  state/time pair has the same number of coefficients; the constant is common.
- Continuous: constant/affine beta(mu), crossed with constant/affine gamma(mu),
  where gamma is measured standardized skewness and M3 = gamma V^(3/2).

Ordinary, unweighted least squares fits measured S, beta, or gamma over each
whole development run. Coefficients are reported in raw state/time units.
Generated trajectories receive only the initial state and fitted laws. The
binary recurrence is T' = T + T(1-T)S; the continuous recurrence is
mu' = mu + beta V and V' = V + beta gamma V^(3/2) - (beta V)^2.
Time rivals use the clock. Invalid states stop generation without clipping;
failed trajectories receive no full-horizon error score. Reported reconstruction
errors exclude the supplied initial state. Missing final neural variance remains
missing, while its observed final mean remains available.

For continuous traits the signed Q error separates exactly into:

```
Q - beta_hat M3_hat
  = (Q - beta M3)                    # linear reweighting
  + beta (M3 - M3_hat)               # moment closure
  + (beta - beta_hat) M3_hat         # selection closure
```

Each relative L2 residual uses the same measured-Q transitions and denominator.
Component norms do not add, and signed errors can cancel. These descriptive
metrics do not establish independent sampling uncertainty or held-out validity.

The output contains four comparison figures (PNG/PDF), a standalone HTML report
with embedded figures and per-run coefficient/error tables, and four CSVs:
`coefficients`, `metrics`, `trajectories`, and signed `residuals`. This milestone
does not fit LLM runs, pool runs, add direct-Q models or exponential tilts, or
select a winning model. Stop for discussion before expanding the comparison.

## Development transfer and multiplicative flux correction

The next agreed comparison tests binary transfer and the single correction
Q = kappa beta M3, using saved controlled data:

```bash
.venv/bin/python -m closure_study.transfer_controlled \
  --table results/closure_study/diagnostics_v3/transitions.jsonl \
  --neural-archive results/trajectory_forecast/retrospective_20260723/simple-theory/main_reproduction_particles/neural/trajectories.npz \
  --output results/closure_study/transfer_v1
```

Binary quadratic state/time laws retain three coefficients each. Own-run curves
and upper roots describe heterogeneity; leave-one-run-out (LOO) fits use the other
whole runs and generate the target from its initial T. Seeds change the environment
as well as training randomness, so coefficient/root shifts need interpretation.

Continuous kappa minimizes sum(Q - kappa beta M3)^2 through the origin, using
measured beta and M3. Window estimates at steps 0:50, 50:100, 100:200, 200:500,
and 500:1000 assess temporal variation; each window's share of sum(beta M3)^2
shows how much it influences the whole-run fit. These window fits are diagnostics,
not extra rollout models. Raw populations at steps 0, 10, 25, 50, 100, 200 locate
the reweighting error by trait value. Weighted bin contributions sum exactly to
Q - beta M3; powers of bin means are not substituted for within-bin moments.

Affine beta(mu) and gamma(mu) are retained. Protocols distinguish own-run kappa,
shared kappa (including the target), LOO kappa with own-run beta/gamma, and full
LOO transfer of beta/gamma/kappa. Only the last transfers every fitted coefficient.
An uncorrected own-run model remains the reference. Kappa changes only Q in the
variance update, never the mean selection or the -(beta V)^2 correction. It is
an empirical flux relation, not a claim that conditional reweighting is linear.

The signed error separates into (Q - kappa beta M3), kappa beta(M3 - M3_hat), and
kappa(beta - beta_hat)M3_hat. The original Q - beta M3 remains separately exported.
The report embeds four figures and per-run tables; six CSVs retain parameters,
metrics, trajectories, flux residuals, window diagnostics, and early trait bins.
These are development transfer checks on inspected runs. No prospective validation,
new training, extra state variables, or general direct-Q closure is introduced.

## LLM closure hypotheses

Test the agreed forms separately on saved output runs and SAE feature/run series:

```bash
.venv/bin/python -m closure_study.fit_llm \
  --table results/closure_study/diagnostics_v3/transitions.jsonl \
  --output results/closure_study/llm_fits_v3
```

Binary fits minimize squared residuals in C = T(1-T)S, using constant,
affine/quadratic state laws and matched time rivals. All settings use this same
flux objective, avoiding division by small binary variance. Fits use only
checkpoints with measured old T and C; missing prevalence is never interpolated.
Boundary observations retain their residuals but supply no coefficient information.
Generated T is scored only against recorded direct prevalence checkpoints.

SAE fits use affine beta(mu), affine standardized skewness gamma(mu), and a
through-origin kappa from measured Q versus beta M3. The uncorrected and corrected
models share beta/gamma; kappa changes only the Q term. Signed residuals separate
flux, moment, and selection errors as in the controlled comparison. All component
fits, residuals, and fit counts use the same finite rows with positive variance;
trajectory scoring retains all measured states. Mean/variance
reconstruction uses old-policy sample panels. Independent direct mean levels,
available in two reruns, have a separate score; historical rounded drift never
supplies missing levels. The final sample-panel mean and variance remain missing.

All coefficients are fitted per run/feature on the whole development series.
Generated paths receive only their initial state and fitted laws; these are
reconstructions, not held-out predictions. Invalid probabilities, negative SAE
means or variance, and nonfinite states stop generation without clipping. Failed
paths have no full-horizon score. Features remain nested within runs, no parameters
are pooled, and sampling uncertainty is not calibrated by these fits.

The report embeds four comparison figures and per-run coefficient/error tables.
Four CSVs save parameters, metrics, generated/observed trajectories, and signed
residuals. Stop to discuss the results before selecting models or adding experiments.

## LLM development transfer and selection diagnostics

```bash
.venv/bin/python -m closure_study.diagnose_llm \
  --table results/closure_study/diagnostics_v3/transitions.jsonl \
  --output results/closure_study/llm_diagnostics_v2
```

For the three above-hook output runs, compare constant S, affine S(T), and affine
S(t). Fit each omitted run's coefficients on the other whole runs, using the same
flux objective and measured old-state/flux pairs as above. Generate the target
from its initial T and score only observed future checkpoints; own-run fits remain
a reference. Pooling is restricted to one setting and trait. These previously
inspected runs support a development transfer check, not prospective validation.

For SAE traits, reuse the per-run affine beta(mu), affine gamma(mu), and kappa
fits. Plot beta against both mu and time, alongside raw C and V. Compare selection
residuals against V and skewness residuals against mu/time across runs. The default
display feature is 25273, the first selected feature in the original pool;
`--feature` explicitly selects another. A fourth figure summarizes residual
correlations for every feature/run. Curved residual patterns can have small
Pearson correlations; these serial, sampled observations do not supply calibrated
uncertainty or establish state dependence rather than schedule dependence.

Four embedded figures and per-run tables accompany five CSVs: binary parameters,
metrics, trajectories, SAE points, and SAE summaries. The existing common valid-row
mask applies to SAE component fits and residuals; raw measured points are retained.
No additional continuous closure is fitted. Stop for discussion before expanding
the closure family or planning new training.

## SAE selection curvature with fixed variance closures

```bash
.venv/bin/python -m closure_study.refine_sae \
  --table results/closure_study/diagnostics_v3/transitions.jsonl \
  --output results/closure_study/sae_selection_v2
```

Compare affine/quadratic beta(mu) and matched affine/quadratic beta(t), all
fitted by minimizing sum(C - V beta_hat)^2 on the common valid rows. The previous
unweighted affine beta(mu) fit remains a reference, separating the objective
change from adding curvature. Covariance fitting is equivalent to weighting
squared beta residuals by V^2; this is not inverse-noise weighting, and it can
improve C fit while worsening beta fit at small V. Both residuals are retained.

Each run/feature keeps its existing affine gamma(mu) and kappa unchanged across
all five comparisons. Only beta uses the clock for time rivals; gamma still uses
the generated mean. There is no pooling. Generate from initial sample-panel mean
and variance, preserve separate direct mean observations and missing final panel
moments, and stop invalid trajectories without clipping or full-horizon scores.

The report contains four figures and per-run tables; four CSVs retain parameters,
metrics, trajectories, and residuals, including beta-fit weights and the complete
variance-increment residual. Q flux and moment residual components are fixed
across selection models, while selection residuals can change or cancel them.
The variance-increment residual compares measured Q - C^2 with the model's
increment at measured states; it is not a direct next-variance measurement.
`--feature` chooses the display feature (default 25273). These are development
reconstructions. Stop for discussion before sharing coefficients or revising the
skewness law or kappa.

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
