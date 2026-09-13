# LLM binary confirmation execution specification

This completes the implementation details of the scientific specification frozen
at `32b632637905980286f7835f493765c246d5635f`. Its coefficients, models, seeds,
measurement budget, scoring windows and failure rules remain unchanged. This
milestone is preparation only. Obtain independent review and commit approval,
then separate authorization before generating confirmation data.

## Runner and measurement

`python llm-experiments/run_binary_confirmation.py --seed <registered-seed>`
executes exactly one registered run. It has no setting, coefficient, measurement,
output-path or seed override beyond choosing one of the three registered seeds.
The command checks the frozen files against the calibration commit, checks all
recorded development data/code hashes, requires committed implementation files,
and saves the implementation commit and source hashes. It refuses a populated
target, including a failed or partial run. There is no resume or overwrite mode.

The existing model loader, LoRA restriction, training update, prompt generator,
answer parser and sequence likelihood calculation are retained. Model/tokenizer
loading uses the frozen Hugging Face revision. Measurement likelihood evaluation
uses batches of 32 responses to bound memory, with unchanged precision and token
scoring. This does not change training batches or prompt-generation grouping.

Training prompt seeds remain `training_seed + 1009*update`. Measurement prompts
use a separate `random.Random(training_seed+200000)` stream: take one 64-bit seed
per checkpoint and pass it to the existing prompt generator. Its 512 generated
prompts each receive two completions. Generation has a persistent Torch CPU/CUDA
stream initialized at `training_seed+100000`, entered only for measurement and
successor scoring. Restore training RNG states and model mode on exit, including
on exceptions. All measurement is in evaluation mode; the registered model has
zero dropout. No additional measurement consumes the training stream.

Save each checkpoint pool immediately, including checkpoint 60, with prompt and
response tokens/text, group and response indices, parsed choice, hint, binary
trait, and old log probability. Save successor probabilities separately, bound
to the exact pool hash and unit-transition endpoints. Training summaries/examples
and the final adapter are retained. An interrupted collection stays on disk and
is reported as incomplete, not reused or silently omitted.

## Exact response-measurement bootstrap

The numerical settings are in `configs/llm_binary_confirmation/execution.json`:
2,000 draws per run, NumPy PCG64 seed `training_seed+300000`, and pointwise 2.5th
and 97.5th percentiles using NumPy's linear quantile interpolation.

For each bootstrap draw and independently at each checkpoint, sample 512 prompt
group indices uniformly with replacement. Keep both responses of a selected
group together. The same indices carry z, old probability and successor
probability together. No response-level resampling within groups, new prompts,
or new completions are generated. Checkpoint pools are sampled independently;
there is no invented pairing of different checkpoints' prompt identities.

For each draw compute T as the response-average binary trait. For each source
checkpoint t, compute raw w=exp(logp_next-logp_old), then
`C_t = mean(w*z) - mean(w)*mean(z)` and `S_t=C_t/[T_t*(1-T_t)]` where defined.
The implementation uses per-group means of z, w and w*z, which exactly reproduce
these response-level statistics because every group has two responses. No
normalization of w, change to covariance bias correction, or coefficient refit.

Use the draw's T_0 to initialize every model. Generate all three trajectories
recursively with the unchanged coefficients, and score them against that same
draw's observed prevalence at checkpoints 5,10,...,60. Compute C/S residuals on
transitions 0,...,59. Use the same draw for all models and for their paired metric
differences. Report all three pairwise comparisons (constant/affine state,
constant/affine time, affine state/affine time); no model is selected here.

Apply the original stopping rules separately to every draw. Save each first
invalid attempt and failure checkpoint. Failed trajectories have no full-horizon
RMSE or terminal error. A raw S value is undefined at T=0 or 1; C remains defined.
Show the number of defined draws for each interval and the number of completed
bootstrap trajectories. Intervals for trajectory scores are conditional on
completion; path intervals are conditional on being defined at that checkpoint.
Do not treat their conditional nature as evidence of reliability. Save bootstrap
T/C, failure states and all model-level metrics so failures remain inspectable.

These intervals quantify response-measurement uncertainty conditional on the
realized trained policy sequence and the frozen development coefficients. They
do not include calibration-coefficient or training-run uncertainty, and are not
simultaneous confidence bands. The three training runs remain the replication
unit. Show each run separately; do not pool checkpoints or bootstrap draws into
an apparent larger training sample. No interval changes the registered model
comparisons, collection budget, scoring windows, failure rules or inclusion rules.

## Analysis and implementation checks

After authorized collection, run
`python -m closure_study.binary_confirmation --output <new-report-directory>`
from the same committed implementation. The readout includes every registered
run, including incomplete collections, model residuals, trajectories, scores,
paired differences and completion/failure states. Numerical tables retain the
pointwise uncertainty details.

Before launch, test the actual training path twice from the same unrelated
scratch seed, with measurement disabled/enabled. Compare sampled training tokens,
summaries and all trainable parameters after every update and in the final
adapter. Predeclare exact equality for random states/tokens and a maximum absolute
parameter tolerance of 1e-7 (zero relative tolerance); report the actual maximum
difference. Also check grouping, transition alignment, probability calculations,
autonomous failure handling, bootstrap pairing and launch refusal conditions.
Reserved confirmation seeds and paths are never implementation-test fixtures.
