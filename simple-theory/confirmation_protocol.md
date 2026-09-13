# Controlled closure confirmation — prepared for approval, not launched

Build a small dynamical model of unrewarded trait change by starting from the
Price equation, identifying what remains unclosed, and using experiments to test
closure assumptions and estimate their constants. Binary and continuous traits
are both central. Interpretation, closure residuals, and generated trajectories
are the evidence; incremental forecasting gains are secondary.

## Claims and comparisons

| Claim under test | Primary comparison | Secondary diagnostics |
| --- | --- | --- |
| A quadratic binary selection law describes fresh training in this environment | Frozen quadratic S(T) versus equally flexible quadratic S(t): measured S residual RMSE and autonomous T RMSE, per run | C residuals, residuals versus state/time, terminal error, completion and upper equilibrium root |
| Binary state dependence survives a changed initialization | The same comparison on the predeclared perturbation arm, with neither refitting nor time alignment | Paired arm differences for three shared training seeds; overlapping observed state ranges and residuals within those ranges |
| The continuous selection and shape laws transfer to new training and initial states | Frozen affine beta(mu) and gamma(mu): measured beta and M3 residuals separately, plus autonomous mean and variance RMSE in each arm | Raw gamma and C residuals; early/late residuals; completion and attempted invalid updates |
| A single flux-amplitude correction transfers | Frozen kappa versus kappa=1 with identical beta/gamma: measured Q residuals and autonomous variance error | Signed flux/shape/selection decomposition, original Q-beta*M3, mean error and terminal variance |

There is no registered continuous time closure. The continuous perturbation tests
the sufficiency and transport of the retained state; it does not by itself prove
that no schedule-dependent description could work. Binary comparisons address
the particular registered schedule rival. A good trajectory cannot rescue a
closure that systematically misses its measured quantity through cancellation.

## Two arms, fixed before new trajectories are inspected

Both systems retain environment seed **290402**, the exact frozen reward/trait
tables and neural features, N=512, K=16, G=8, hidden-quality gamma=0.5,
p=0.3, alpha=1, and batch size 64. Binary: 2,000 updates, eta=0.3.
Continuous: 1,000 updates, Adam lr=0.02, d=32, N_eval=512. The continuous trait
is the existing standardized score; it can be negative. Do not impose the
nonnegative SAE mean or SAE moment constraint on it.

* Replication: five fresh training seeds, original uniform initial policy.
* Perturbation: the first three of those seeds, initial quality-logit tilt +0.5.
  Binary initial logits are 0.5 times the existing latent quality z. Neural
  initial head has w[0]=0.5 and every other coordinate zero, giving logits
  0.5 times the existing train-standardized quality coordinate. The projected-out
  trait coordinate stays excluded. Adam starts with empty state in both arms.

This changes only the initial policy, with no warm-up training or reward change.
The two tilts use their systems' existing quality coordinates; they need not
produce the same shift in trait units. Magnitude and direction are not tuned
after inspecting trajectories. Three paired seeds couple random draws across
arms; they are three paired comparisons, not six independent replicates.
The five seeds are 2026091301–2026091305 for binary and
2026091401–2026091405 for continuous. Environment generation and training
randomness are separated after the frozen environment is constructed.

The state model receives only the new T0 or (mu0,V0). Every subsequent prediction
uses its own generated state. Time rivals additionally receive the integer update
index, starting at zero in both arms: no fitted time shift or state resets.

## Exact frozen coefficients

The executable source of these values is [configs/confirmation.yaml](configs/confirmation.yaml).
They are copied without refitting from `results/closure_study/transfer_v1/parameters.csv`,
seed-290402 `state_own`, `time_own`, and `corrected_own` rows.
Coefficients are listed in ascending power order; t counts individual updates.

| Law | Numerical coefficients |
| --- | --- |
| S(T)=c0+c1*T+c2*T^2 | -0.004732950770631918, 0.026611179363419956, -0.030248698205804156 |
| S(t)=c0+c1*t+c2*t^2 | 0.0008039589955753542, 5.324154303000465e-07, -5.059417704939305e-10 |
| beta(mu)=b0+b1*mu | 0.006318034798833055, -0.00801987797791238 |
| gamma(mu)=g0+g1*mu | 2.3743182287192925, -1.5847566562120066 |
| kappa | 0.7781567241571242; reference 1.0 |

The original objectives remain unchanged: binary least squares in S; continuous
least squares in raw beta and gamma, then through-origin least squares in Q
against measured beta*M3. This milestone copies their fitted constants and does
not optimize any objective. It does not import the later SAE fitting objectives.

## Measurement and scoring

Enumerate the full fixed evaluation support every update, including the terminal
state. Compute float64 moments and C,Q from successive population weights, using
the existing normalization and moment routines. Keep the independent directly
enumerated mean/variance changes to check the Price identities. No response
sampling or measurement bootstrap is necessary for these finite tables.

Retain the existing recurrences and stopping rules: binary T must stay in [0,1];
continuous variance must stay nonnegative; all generated states must be finite.
Never clip. Save the first attempted invalid update; failed paths have no
full-horizon RMSE and remain in run counts. Continuous means are standardized,
so a negative mean alone is not failure.

Score binary S and C residuals over t=0..1999, and T over t=1..2000.
For continuous selection use t=0..999. For M3/Q decomposition and variance RMSE
retain the development-comparable window (transitions 0..998, states 1..999);
report the now-available final transition and terminal variance separately.
Mean RMSE uses states 1..1000. Record M3 residuals and raw gamma separately.
Continuous early/late diagnostics use the already employed first 100 updates
versus the remainder. Show residuals against state and time in both arms.

Report every run, paired model differences, and the three paired arm comparisons;
do not treat updates as independent replicates. Do not set a post hoc success
threshold or refit after seeing confirmation. Completion alone is not accuracy.

## Implementation and launch boundary

Before approval: reproduce the old seed-290402 trajectories; check separation of
environment/training randomness, the changed initial policy and unchanged tables,
the exact moment identities, and frozen prediction behavior on small fixtures.
Do not run reserved training streams, or preview the perturbed confirmation
trajectory, as an implementation check. Tiny fixtures use unrelated seeds and
tables. Initial moments can be checked without training.

Obtain independent review and stop for commit approval. No confirmation launch
is part of this milestone. After approval, each invocation writes one arm/seed
to a new directory, with the config, measured transitions, generated trajectories,
residuals, and scores. No experiment-management framework is needed.

## Following systems and separate SAE gate

LLM binary confirmation remains next after the controlled systems: the existing
above-hook Qwen2.5-0.5B setting, three fresh training runs, constant/affine state
and affine time laws. Freeze its pooled development coefficients before launch.

SAE new-feature confirmation remains separately gated. First inspect the existing
15 selected/control features using deterministically chosen top-response and
token contexts, and report predeclared activation correlations and redundancy
clusters. Features within a run are not independent replications. The scratch
semantic/redundancy report will inform whether this stress test is interpretable
enough to retain. Do not change the closure families in this milestone.

If retained, finalize the prospective dictionary's availability and genuinely
untouched status before opening it; layer 11 remains an unverified candidate.
Freeze discovery, feature identities and calibration before fresh longitudinal
outcomes. Consider a fixed high-N measurement tier at predetermined transitions
in that separately reviewed protocol; do not launch it, increase N reactively,
or exclude features based on confirmation outcomes. The earlier scratch SAE
proposal is provisional and is not a launch authorization.
