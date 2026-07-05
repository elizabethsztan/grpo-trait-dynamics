# How we compute the trait T and the importance weight ω

Reference for Phase 3 (`experiments/run_price_eval.py`) and the modules it calls.
All quantities are evaluated on the **held-out** split `D_eval`, with rollouts
sampled **independently** of the GRPO gradient rollouts.

## 1. Trait score of a single rollout — `s(x, a)`

For a prompt `x` and a sampled completion `a` (`src/trait.py::collect_feature_scores`):

1. Teacher-force the exact token sequence `[x ; a]` through the frozen base model in
   one forward pass.
2. Read the **post-block residual stream at layer L=12** (the module output
   `model.layers.12`) — this is the exact activation the Qwen-Scope SAE was trained on.
3. Encode the **completion tokens only** with the frozen TopK SAE
   (`src/sae.py`): `f = TopK_ReLU(h @ W_encᵀ + b_enc)`, k = 100.
4. Score each SAE feature `m` as the **max activation over completion tokens**:

$$ s_m(x, a) \;=\; \max_{\tau \in a} f_{m}(x, a)_\tau \;\ge 0 $$

Note `s ≥ 0` always (TopK + ReLU). This matters below.

Because LoRA lives only on layers **> L**, `h_L` — and therefore `s` — is **frozen**:
`s` is identical under π_t and π_{t+1} for the same `(x, a)`. (Asserted at train time.)

## 2. Trait expectation and its drift — `T_t`, `ΔT_direct`

`T_t` is the average trait under the policy at step `t`:

$$ T_t \;=\; \mathbb{E}_{x\sim D_{\text{eval}},\, a\sim \pi_t(\cdot\mid x)}\big[s(x,a)\big]
\;\approx\; \frac{1}{N_{\text{dir}}}\sum_{i=1}^{N_{\text{dir}}} s(x_i, a_i),\quad a_i\sim\pi_t $$

The **direct** drift draws *fresh, independent* rollouts from each policy
(`direct_samples` each) and differences the means:

$$ \widehat{\Delta T}_{\text{direct}} \;=\; \hat T_{t+1} - \hat T_t $$

This is the "ground-truth" line the Price estimate is compared against.

## 3. Importance weight — `ω`

For a rollout `(x, a)` sampled from **π_t**, ω is the per-sequence policy ratio
(`src/logprob.py::sequence_logprob`, called under each checkpoint's adapter):

$$ \omega(x,a) \;=\; \frac{\pi_{t+1}(a\mid x)}{\pi_t(a\mid x)}
\;=\; \exp\!\big(\log\pi_{t+1}(a\mid x) - \log\pi_t(a\mid x)\big) $$

where the sequence log-prob is the **sum of token log-probs over the exact sampled
action span** (including the terminating EOS iff generation stopped on EOS):

$$ \log\pi(a\mid x) \;=\; \sum_{\tau\in a} \log \pi(a_\tau \mid x, a_{<\tau}) $$

Both log-probs use the **same** token span, so ω is the ratio for the identical action.
Sanity identity: on samples from π_t, `E[ω] = 1` (probability is conserved) — we log
`mean_omega` as a bug-check, **not** as a measure of selection.

## 4. The Price estimate of the drift

Sample `(x_i, a_i) ~ p_t` (i.e. `x_i ∼ D_eval`, `a_i ∼ π_t`), compute `s_i` and `ω_i`.
Two equivalent-in-expectation estimators (both currently logged per row):

$$
\underbrace{\widehat{\Delta T}_{\text{price}} \;=\; \overline{\omega s} - \bar s}_{\texttt{"price"}}
\qquad
\underbrace{\widehat{\Delta T}_{\text{cov}} \;=\; \overline{\omega s} - \bar\omega\,\bar s
\;=\; \widehat{\mathrm{Cov}}(\omega, s)}_{\texttt{"cov"}}
$$

The theory says `ΔT = Cov(ω, s)` — selection = ω covarying with s (probability flowing
toward high-trait rollouts), **not** the mean of ω.

## 5. The problem we found, and what we're changing

`"price"` assumes `ω̄ = 1` exactly. In finite samples `ω̄ ≈ 1.05`, and the two forms
differ by `(ω̄ − 1)·s̄`. Because **`s ≥ 0` ⇒ `s̄ > 0` for every feature**, that leaves a
**systematic positive bias in `"price"` on every feature** — the overshoot seen in
`grid_price` (orange above blue for positive features, under-magnitude for negative).

Empirically (pilot, lr=1e-4, pooled over tracked features):

| estimator | corr vs direct | slope | mean signed bias |
|---|---|---|---|
| `price` (`ω̄` assumed 1) | 0.84 | 0.78 | **+0.018** |
| `cov`  (`ω̄`-corrected)  | **0.92** | **0.89** | **+0.002** |

### Changes
1. **Use the covariance form `cov` as the primary Price estimate** in plots and
   analysis (it drops the `ω̄`-bias and is the quantity the theory actually predicts).
   `price` stays logged for reference.
2. **Plotting:** plot `cov` vs `direct` with a residual panel, and label each feature
   with its ρ (reward-correlation) + group by sign, so tracking is legible.
3. **Convergence check (the real validation):** the pilot's small `N` leaves genuine
   estimator noise (and the OLS slope is attenuated because `direct` is itself noisy).
   The definitive test is whether `cov → direct` as `N` grows, which needs the full
   run's larger budgets (`price_samples [128,512,2048]`, `direct_samples 8192`) — or a
   cheap single-checkpoint sweep at `N = 256/1024/4096`.

### Not changing (already correct)
- Held-out `D_eval`, rollouts independent of the gradient step.
- Frozen trait (LoRA strictly downstream of L); EOS included in the action span.
- `mean_omega`/ESS diagnostics as bug/reliability checks.
