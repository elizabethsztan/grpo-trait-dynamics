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

1. **Location & module:** both policy classes live in one module. Rename
   `src/tabular_policy.py` → `src/policy.py`, holding `class TabularPolicy` (the current
   `Policy`, renamed) and the new `class NeuralPolicy`. No separate `neural_policy.py`. The
   empty `simple-nn/` directory at the repo root is deleted.
2. **Unified runner:** **no new `run_neural.py`.** `run_experiment.py` gains a
   `--mode {tabular,neural}` CLI flag that only chooses which `build_*` function constructs
   the policy. The step loop already calls just `get_T()` / `expected_reward()` /
   `grpo_step(...)`, so it stays generic over both policy classes.
3. **Single shared config file:** one `configs/config.yaml` holds all blocks; `--mode`
   selects which are read. The run directory is suffixed by mode (`results/{name}/{mode}`)
   so the two modes never clobber each other from one file.
4. **Update rule:** **PyTorch autograd** — `w` is a real trainable parameter trained with a
   torch optimizer on the GRPO loss. Matches the outline's "update via backprop" framing and
   is the natural stepping stone to deeper NN policies later.
5. **Default optimizer:** **Adam** (robust default), with SGD available as a config switch.
6. **Eval protocol:** a **held-out eval set** — separate frozen `train` and `eval`
   activation tables. Because `w` is shared across all prompts the neural policy genuinely
   generalises, so "trait rises on prompts never trained on" is a real result.
7. **No random rotation in this cut** — `h = h0` always, probes are the coordinate vectors
   `e_1`, `e_2`. Rotation is deferred until the core result works (§8).
8. **Tests:** not part of this first cut (per user). Optional later sanity checks in §8.

## 3. Files (all inside `simple-theory/`)

- `src/policy.py` — **renamed** from `tabular_policy.py`. Contains `TabularPolicy` (the
  current `Policy`, renamed) and the new `NeuralPolicy` (PyTorch). The shared `sigmoid`
  helper stays at module level.
- `run_experiment.py` — **modified**: update the import to `policy`/`TabularPolicy`; add
  `--mode {tabular,neural}`, a `build_neural_policy` alongside the existing `build_policy`,
  generic collection of optional extra metrics, and the mode-suffixed output dir.
- `run_sweep.py` — **unchanged**: it imports `from run_experiment import run_config,
  cfg_value` and never references the policy class directly, so the rename does not touch it.
- `configs/config.yaml` — **extended** with a `NeuralPolicyConfig` block and neural-only
  training fields; same file drives both modes.
- `pyproject.toml` — add `torch` to dependencies.
- Delete the empty `simple-nn/` directory at the repo root.

## 4. `NeuralPolicy` class

### 4.1 State

**Frozen** (built in `init_env`):
- `h_train` shape `(N_train, K, d)`, `h_eval` shape `(N_eval, K, d)` — activation tables.
- `q_train`, `q_eval` — reward probabilities `σ(α z)`.
- `s_eval` — continuous trait scores; `z_eval` — quality scores (diagnostics).
- `v`, `u_z` shape `(d,)` — frozen trait and quality probes (`e_2`, `e_1`).

**Trainable:**
- `w` shape `(d,)`, initialised to **zeros** (gives uniform `π_0 = 1/K`), wrapped in a torch
  optimizer (Adam by default).

### 4.2 Environment generation (Version B, outline §2.4–2.5)

For each table of size `(n, K)`:

- `z ~ N(0,1)`; `q = σ(α z)`. **Reward uses the raw `z`.**
- `m = μ_p + γ z + sqrt(1-γ²) ξ`, `ξ ~ N(0,1)`, with `μ_p = Φ⁻¹(p_trait)`.
- `s_raw = ReLU(m)`.
- **Standardise `z` and `s` using train-set statistics only** (proper held-out protocol:
  eval is standardised with train mean/std, never its own).
- Build `h0 = [z_std, s_std, ε_1, …, ε_{d-2}]` with `ε_j ~ N(0,1)`, so that
  `u_z = e_1`, `v = e_2` give `u_z^T h = z_std` and `v^T h = s_std`. In this cut
  `h = h0` (no rotation).

The trait used in all metrics is the standardised `s_std`, and `v^T h` equals it by
construction.

### 4.3 Interface (mirrors `TabularPolicy`)

- `get_pi(split)` — logits `h @ w`, softmax over the `K` axis.
- `get_T()` — expected internal trait on **eval**: `mean_x Σ_a π(a|x) s_eval(x,a)`.
- `expected_reward()` — `mean_x Σ_a π(a|x) q_eval(x,a)` on eval.
- `grpo_step(batch_size, eta=None, eps, price_check)` — same signature shape as
  `TabularPolicy`; returns the price increment (or `None`). The neural step uses
  `lr`/optimizer internally; the `eta` arg is accepted-and-ignored so the runner loop is
  identical across modes.
- `extra_metrics()` — returns a dict of neural-only diagnostics for the current policy:
  `{"Z": ..., "A_active": ..., "w_dot_uz": ..., "w_dot_v": ...}`. `TabularPolicy` either
  lacks this method or returns `{}`, so `run_single` collects extras generically.

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
before the optimizer step and `π_{t+1}` after; `grpo_step(price_check=True)` returns the
increment.

## 5. Runner (`run_experiment.py`, modified)

- New CLI: `--mode {tabular,neural}` (default `tabular`), alongside `--config`.
- `build_policy` (tabular, unchanged apart from the renamed class) and a new
  `build_neural_policy` construct the policy; `main` dispatches on `--mode`.
- `run_single` / `run_config` stay generic: they collect the core curves (`T_t`, `R_t`, price
  increments) for both modes, plus any keys returned by `policy.extra_metrics()` each step
  (empty for tabular). Aggregation (mean ± sem over `num_runs` seeds) is unchanged.
- Output dir becomes `results/{name}/{mode}` to keep the two modes separate when driven from
  one config. Saves `metrics.json` (floats rounded to 4 dp) and copies the config, matching
  Part 1 conventions.
- Plots: reuse `plot_trait`, `plot_trait_and_reward`, `plot_price_check`. In neural mode add
  one small plot of the projections `w_t^T u_z` and `w_t^T v`.

Note: `run_sweep.py` (the `gamma × p` grid, lists in the config) is the existing tabular
sweep workflow; it imports only from `run_experiment` and is untouched by this change.

## 6. Config (`configs/config.yaml`, extended)

Single-run, scalar-valued. Neural-only keys are read only under `--mode neural`.

```yaml
PolicyConfig:
  N: 512                 # = N_train
  K: 16
  G: 8
  mode: hidden_quality   # reward story; neural always uses hidden_quality

HiddenQualityConfig:
  gamma: 0.5
  p: 0.3                 # = p_trait, P(s > 0)
  alpha: 1.0

NeuralPolicyConfig:      # read only when --mode neural
  N_eval: 512
  d: 32

TrainConfig:
  steps: 2000
  batch_size: 64
  num_runs: 5
  seed: 290402
  eta: 0.3               # tabular learning rate
  lr: 0.1                # neural: optimizer learning rate
  optimizer: adam        # neural: adam | sgd
  beta_kl: 0.0           # neural: KL-to-uniform penalty

OutputConfig:
  results_dir: results
  name: hidden_quality

PriceCheckConfig:
  enabled: true
```

(A sweep config still sets `gamma`/`p` as lists for `run_sweep.py`, exactly as today.)

## 7. Build order

1. Add `torch` to `pyproject.toml`; delete `simple-nn/`.
2. Rename `tabular_policy.py` → `policy.py`, rename `Policy` → `TabularPolicy`, update the
   import in `run_experiment.py` (only importer). Confirm Part 1 still runs.
3. `NeuralPolicy.init_env` (held-out data generation, standardised on train stats).
4. `get_pi` and metrics (`get_T`, `expected_reward`, `extra_metrics`).
5. `grpo_step` + exact price check.
6. Wire into `run_experiment.py`: `--mode`, `build_neural_policy`, generic extras, output
   dir suffix, projection plot. Extend `configs/config.yaml`.
7. Smoke run small (`N=64`, `steps=100`), then the full config.

## 8. Deferred (YAGNI for the first cut)

- **Random rotation** of the activation space (`h = Q h0`, `v = Q e_2`, `u_z = Q e_1`, fixed
  orthogonal `Q`) — add after the unrotated result works; remember to check the row/column
  convention (`Qv` vs `Qᵀv`).
- A `(γ, p_trait)` sweep runner mirroring `run_sweep.py`.
- The KL ablation (`β_KL > 0`).
- Optional sanity checks (if wanted later): `v^T h == s_std` and `u_z^T h == z_std`,
  `w_0=0 ⇒ π_0=1/K`, and the exact one-step Price identity
  `T_{t+1}-T_t == Cov_{p_t}(ω,s)`.

## 9. Expected result

For `γ > 0`: `R_t` rises (GRPO optimises reward via `z`) and `T^int_t` rises with it, even
though the reward function never reads the trait — the trait is amplified purely because it
co-occurs with quality. The exact Price check should show `T_t - T_0` matching the cumulative
`Cov_{p_t}(ω_t, s)`. The effect should hold on the held-out eval prompts, demonstrating the
neural policy generalises the selection.
