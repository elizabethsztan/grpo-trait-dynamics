# Part 2: Frozen-NN Policy with Continuous Activation Trait — Design

**Date:** 2026-06-22
**Status:** Approved, ready for implementation planning
**Outline reference:** `simple-theory/outline.md` §2

## 1. Purpose

Reproduce the Part 1 selection result (GRPO amplifies a reward-correlated trait) in the
smallest possible *neural* policy. The move is:

```
tabular logit table + binary trait table   ->   neural policy head + frozen activation trait
```

Because the hidden activations `h` and the trait probe `v` are frozen, `Δs_t(x,a) = 0`, so
this experiment isolates the **selection term** of the Price equation. It does not yet test
representation drift or the transmission term.

## 2. Key decisions (settled during brainstorming)

1. **Location:** a new `NeuralPolicy` class living inside the existing `simple-theory`
   project (not a separate sub-project, not an extension of `Policy`). The empty
   `simple-nn/` directory at the repo root is deleted.
2. **Update rule:** **PyTorch autograd** — `w` is a real trainable parameter trained with a
   torch optimizer on the GRPO loss. This matches the outline's "update via backprop"
   framing and is the natural stepping stone to deeper NN policies later.
3. **Eval protocol:** a **held-out eval set** — separate frozen `train` and `eval`
   activation tables. Because `w` is shared across all prompts the neural policy genuinely
   generalises, so "trait rises on prompts never trained on" is a real result.
4. **Default optimizer:** **Adam** (robust default), with SGD available as a config switch.
5. **Tests:** not part of this first cut (per user). Listed as an optional later sanity
   check in §8.

## 3. Files (all inside `simple-theory/`)

- `src/neural_policy.py` — new `NeuralPolicy` class (PyTorch), mirroring the `Policy`
  interface so the runner and plotting reuse cleanly.
- `run_neural.py` — runner; imports and reuses the plotting functions from
  `run_experiment.py`.
- `configs/neural_config.yaml` — its own config blocks.
- `pyproject.toml` — add `torch` to dependencies.
- Delete the empty `simple-nn/` directory at the repo root.

## 4. `NeuralPolicy` class

### 4.1 State

**Frozen** (built in `init_env`):
- `h_train` shape `(N_train, K, d)`, `h_eval` shape `(N_eval, K, d)` — activation tables.
- `q_train`, `q_eval` — reward probabilities `σ(α z)`.
- `s_eval` — continuous trait scores; `z_eval` — quality scores (diagnostics).
- `v`, `u_z` shape `(d,)` — frozen trait and quality probes.

**Trainable:**
- `w` shape `(d,)`, initialised to **zeros** (gives uniform `π_0 = 1/K`), wrapped in a torch
  optimizer.

### 4.2 Environment generation (Version B, outline §2.4–2.5)

For each table of size `(n, K)`:

- `z ~ N(0,1)`; `q = σ(α z)`. **Reward uses the raw `z`.**
- `m = μ_p + γ z + sqrt(1-γ²) ξ`, `ξ ~ N(0,1)`, with `μ_p = Φ⁻¹(p_trait)`.
- `s_raw = ReLU(m)`.
- **Standardise `z` and `s` using train-set statistics only** (proper held-out protocol:
  eval is standardised with train mean/std, never its own).
- Build `h0 = [z_std, s_std, ε_1, …, ε_{d-2}]` with `ε_j ~ N(0,1)`, so that
  `u_z = e_1`, `v = e_2` give `u_z^T h = z_std` and `v^T h = s_std`.

**Optional random rotation** (off by default; enable after the unrotated version works):
sample a fixed orthogonal `Q ∈ R^{d×d}`, set `h = Q h0`, `v = Q e_2`, `u_z = Q e_1`.

Note: the trait used in all metrics is the standardised `s_std`, and `v^T h` equals it by
construction — consistent whether rotated or not.

### 4.3 Interface (mirrors `Policy`)

- `get_pi(split)` — logits `h @ w`, softmax over the `K` axis.
- `get_T()` — expected internal trait on **eval**: `mean_x Σ_a π(a|x) s_eval(x,a)`.
- `expected_reward()` — `mean_x Σ_a π(a|x) q_eval(x,a)` on eval.
- Diagnostics: `get_Z()`, `get_A_active()` (mass on `s>0` actions), and the projections
  `w @ u_z`, `w @ v`.

All metrics are computed exactly by enumeration over the eval table.

### 4.4 `grpo_step`

1. Sample a batch of `B` train prompts.
2. Compute `π` over all `K` actions for those prompts (with grad).
3. Sample `G` actions per prompt from the detached `π`.
4. Sample Bernoulli rewards from `q_train`.
5. Group-normalised advantages `A = (r - r̄) / (σ + ε)`; zero the group if `σ ≈ 0`.
6. Loss `L = -(1/BG) Σ A log π(a|x)`, plus optional `β_KL · KL(π || π_0)` (default `β_KL=0`,
   `π_0` uniform).
7. `loss.backward()` then `optimizer.step()` — updates only `w`.

### 4.5 Price-equation check (exact)

Because `w` is global, every eval `(x,a)` has a well-defined `ω_t = π_{t+1}/π_t`. So compute

```
Cov_{p_t}(ω_t, s) = E_{p_t}[ω_t s] - E_{p_t}[ω_t] E_{p_t}[s],   p_t(x,a) = (1/N_eval) π_t(a|x)
```

by **enumeration** over the whole eval table — no Monte-Carlo estimator like Part 1. This is
an exact test of the §2.9 identity `T_{t+1} - T_t = Cov_{p_t}(ω_t, s)`. Capture `π_t` on eval
before the optimizer step and `π_{t+1}` after.

## 5. Runner (`run_neural.py`)

- Parse config, build `NeuralPolicy`, `init_env()`.
- Loop over `steps`, collecting per-step curves for `T^int_t`, `R_t`, `Z_t`,
  `A^active_t`, `w_t^T u_z`, `w_t^T v`, and the price increments.
- Repeat over `num_runs` seeds; aggregate mean ± sem (same shape as `run_config`).
- Save `metrics.json` (floats rounded to 4 dp) and copy the config into the run dir, matching
  Part 1 conventions.
- Plots: reuse `plot_trait`, `plot_trait_and_reward`, `plot_price_check` from
  `run_experiment.py`; add one small plot for the two projections `w^T u_z`, `w^T v`.

## 6. Config (`configs/neural_config.yaml`)

```yaml
NeuralPolicyConfig:
  N_train: 512
  N_eval: 512
  K: 16
  G: 8
  d: 32
  rotate: false

NeuralEnvConfig:
  gamma: 0.5
  p_trait: 0.3
  alpha: 1.0

TrainConfig:
  steps: 2000
  batch_size: 64
  optimizer: adam     # or sgd
  lr: 0.1
  beta_kl: 0.0
  num_runs: 5
  seed: 290402

OutputConfig:
  results_dir: results
  name: neural_hidden_quality

PriceCheckConfig:
  enabled: true
```

## 7. Build order

1. Add `torch` to `pyproject.toml`; delete `simple-nn/`.
2. `NeuralPolicy.init_env` (data generation + optional rotation).
3. `get_pi` and metrics (`get_T`, `expected_reward`, diagnostics, projections).
4. `grpo_step` + exact price check.
5. Runner + config + plots.
6. Smoke run small (`N=64`, `steps=100`), then the full config.

## 8. Deferred (YAGNI for the first cut)

- A `(γ, p_trait)` sweep runner mirroring `run_sweep.py`.
- The KL ablation (`β_KL > 0`).
- Optional sanity checks (if wanted later): `v^T h == s_std` and `u_z^T h == z_std`
  (unrotated and rotated), `w_0=0 ⇒ π_0=1/K`, and the exact one-step Price identity
  `T_{t+1}-T_t == Cov_{p_t}(ω,s)`.

## 9. Expected result

For `γ > 0`: `R_t` rises (GRPO optimises reward via `z`) and `T^int_t` rises with it, even
though the reward function never reads the trait — the trait is amplified purely because it
co-occurs with quality. The exact Price check should show `T_t - T_0` matching the cumulative
`Cov_{p_t}(ω_t, s)`. The effect should hold on the held-out eval prompts, demonstrating the
neural policy generalises the selection.
