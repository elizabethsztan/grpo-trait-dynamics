# Price-Equation / Selection-Differential Check — Design Spec

**Date:** 2026-06-20
**Status:** Approved (design), pending implementation plan
**Component:** `simple-theory` tabular GRPO simulation

## 1. Motivation

In the tabular model the Price decomposition is an exact algebraic identity:

$$T_{t+1} - T_t = \text{Cov}_{p_t}(\omega_t, s), \qquad \omega_t(x,a) = \frac{\pi_{t+1}(a\mid x)}{\pi_t(a\mid x)}, \qquad p_t(x,a) = \tfrac1N \pi_t(a\mid x).$$

It holds because the trait is frozen ($\Delta s = 0$, no transmission term) and $\mathbb{E}_{p_t}[\omega_t] = 1$ exactly. Computed from the full $\pi$ tables the two sides agree to machine precision, so as a plot it would only be a unit test.

The reason to build it is the **NN version**, where the situation is different:

- $(x,a)$ cannot be enumerated, so there is no exact $T_t$ and no exact Cov.
- Both must be **estimated from sampled rollouts**: per-token logprobs under $\pi_t$ and $\pi_{t+1}$ give $\omega$ on a sampled sequence, and the trait $s$ is measured on the sampled output.
- The open question becomes: **does the sampled selection differential $\widehat{\text{Cov}}(\omega,s)$ predict the observed trait drift?** If yes, selection alone explains the drift. If a gap remains, transmission / KL-to-reference / clipping / off-policy staleness is moving the trait, and the residual is itself a finding.

The artifact we port to the NN is therefore an **estimator**, not an identity. The tabular model is the one place with both the estimator and exact ground truth, so it is the rig to validate the estimator before running blind on the NN.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Estimator sampling | **Reuse the GRPO rollout batch** | Most faithful to the NN, which will only have the rollout batch; zero extra sampling cost. Fixes the sample size at $B\times G$ per step (no $M$ to tune). |
| Deliverable | **Plot + citable summary number** | A `price_check.pdf` diagnostic plus a `metrics.json` entry quantifying "selection accounts for X% of the trait drift." |
| Exact-identity sanity plot | **Not included** | The estimator-vs-truth plot is the point; the exact identity is a near-trivial unit test we skip. |
| Code architecture | **Approach A** — `grpo_step` returns the per-step selection increment under an opt-in flag | Smallest surface change; keeps the faithful "reuse the rollout batch" semantics literally in the one place that holds the batch; normal runs untouched. |

## 3. The estimator and its scaling

Over the $B\times G$ sampled $(x,a)$ pairs this step, compute the sample covariance with the full formula (do **not** assume $\mathbb{E}[\omega]=1$, since the sample mean of $\omega$ will not be exactly 1):

$$\widehat{\text{Cov}}_t = \overline{\omega s} - \bar\omega\,\bar s, \qquad \omega_i = \frac{\pi_{t+1}(a_i\mid x_i)}{\pi_t(a_i\mid x_i)}, \quad s_i = s(x_i, a_i).$$

Because a tabular logit update only moves the $B$ batched prompts ($\omega = 1$ on all others), the full-table trait change relates to the batch covariance by a fixed factor:

$$\Delta T_t^{\text{full}} = T_{t+1} - T_t = \frac{1}{N}\sum_{x\in\text{batch}}\sum_a \big(\pi_{t+1}-\pi_t\big)(a\mid x)\,s(x,a) = \frac{B}{N}\,\text{Cov}^{\text{batch}}_t.$$

Derivation sketch: with $\mathbb{E}_{\text{batch}}[\cdot] = \frac1B\sum_{x\in\text{batch}}\sum_a \pi_t(a\mid x)(\cdot)$ we have $\mathbb{E}_{\text{batch}}[\omega]=1$ exactly (each $\pi_{t+1}(\cdot\mid x)$ sums to 1), so $\text{Cov}^{\text{batch}} = \mathbb{E}_{\text{batch}}[\omega s] - \mathbb{E}_{\text{batch}}[s] = \frac1B\sum_{x\in\text{batch}}(\text{per-prompt }\Delta)$, and $\Delta T^{\text{full}} = \frac1N\sum_{x\in\text{batch}}(\text{per-prompt }\Delta) = \frac{B}{N}\,\text{Cov}^{\text{batch}}$.

The Monte-Carlo estimate $\widehat{\text{Cov}}_t$ over the sampled actions approximates $\text{Cov}^{\text{batch}}_t$. So the **per-step predicted increment** that `grpo_step` returns is:

$$\widehat{\Delta T}_t = \frac{B}{N}\,\widehat{\text{Cov}}_t.$$

## 4. Architecture (Approach A)

### `src/tabular_policy.py` — `grpo_step`

- Add an opt-in parameter (default off) so existing runs are byte-for-byte unchanged and pay no cost.
- When enabled, after the logit update compute, using state already local to the method (`batch_idx`, `actions`, `pi_batch`):
  - $\pi_{t+1}$ on the batch = softmax of the updated `self.logits[batch_idx]`.
  - $\omega$ on the sampled actions = $\pi_{t+1}(a_i\mid x_i) / \pi_t(a_i\mid x_i)$ (gather with `take_along_axis`), shape $B\times G$.
  - $s$ on the sampled actions = gather from `self.s[batch_idx]`, shape $B\times G$.
  - Flatten, compute $\widehat{\text{Cov}}_t = \overline{\omega s} - \bar\omega\,\bar s$.
  - Return $\widehat{\Delta T}_t = (B/N)\,\widehat{\text{Cov}}_t$ as a float.
- When disabled, return `None`. The trait/reward curves do not depend on this return value.

### `run_experiment.py`

- `run_single`: when the price check is enabled, pass the flag to `grpo_step` and accumulate the returned increments into a `selection_increments` array of length `steps`. Return it alongside the existing `trait_curve`/`reward_curve`.
- Observed cumulative drift is already available: `trait_curve - trait_curve[0]`.
- Predicted cumulative drift: `np.cumsum(selection_increments)` (prepend 0 to align with the step axis).
- Residual: `observed - predicted`.
- Average across seeds as we already do for trait/reward; carry SEM bands.

### New plotting function — `plots/price_check.pdf` (+ `.png`)

Two stacked panels, existing paper style (serif, top/right spines off), SEM shading across seeds, fills drawn before lines:

- **Top:** observed $T_t - T_0$ vs predicted $\sum_{\tau\le t}\widehat{\text{Cov}}_\tau$ (two labelled lines; should overlap). x-axis $t$, y-axis "Cumulative trait change".
- **Bottom:** residual (observed − predicted), flat near zero if selection explains the drift. Horizontal reference line at 0.

### `metrics.json`

Extend with:
- `selection_explained_frac = predicted_total / observed_total` (final-step ratio),
- `final_rel_error = |residual_final| / |observed_total|`,
- the per-step `observed_cumulative` and `predicted_cumulative` mean/SEM series (rounded, consistent with existing entries).

### Config

New optional block; absence preserves current behaviour:

```yaml
PriceCheckConfig:
  enabled: true
```

`run_experiment.py` reads `cfg.get("PriceCheckConfig", {}).get("enabled", False)`.

## 5. Success criteria

- With `enabled: false` (or block absent), runs are unchanged from current behaviour.
- With `enabled: true`, the top panel shows predicted overlapping observed within the SEM band, and the residual panel is flat near zero.
- `selection_explained_frac` is close to 1 (the headline number); `final_rel_error` is small.
- Runs cleanly on both modes (`trait_drives_reward`, `hidden_quality`).

## 6. Caveats to carry to the NN

- **Support disanalogy.** In the tabular model a logit step only moves the batched prompts, so $\omega \neq 1$ lives entirely on the batch. On the NN a gradient step moves shared weights, so $\omega \neq 1$ on *all* prompts, including unsampled ones. The estimator *mechanics* (sample $(x,a)$, get $\omega$ from the logprob ratio, measure $s$, take the covariance) port directly; the $B/N$ scaling does **not** — on the NN, $\Delta T$ and Cov are measured on the same rollout and self-consistently scaled, so the $B/N$ factor must not be copied blindly.
- **No KL term here.** The clean identity relies on frozen $s$ and no KL-to-reference penalty. Real GRPO may add one, so on the NN the estimator might not fully close the gap; that residual is expected and informative, not a bug.
- **Variance.** Most sampled $(x,a)$ on the batched prompts still contribute little selection signal, which inflates estimator variance; this is what would drive the sample-size requirement on the NN, where the rollout budget is fixed.

## 7. Out of scope (YAGNI)

- The exact-table identity sanity plot.
- A fresh-i.i.d.-batch estimator and any sample-size ($M$) sweep.
- Per-step (non-cumulative) scatter plots.
- Decomposing Cov by traitful vs non-traitful actions (possible future teaching plot).
