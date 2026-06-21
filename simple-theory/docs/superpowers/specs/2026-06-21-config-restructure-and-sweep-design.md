# Config Restructure + Version-B `p` Parameter + (gamma x p) Sweep — Design Spec

**Date:** 2026-06-21
**Status:** Approved (design), pending implementation plan
**Component:** `simple-theory` tabular GRPO simulation
**Relation to other specs:** Independent of the Price-equation check spec (`2026-06-20-price-equation-check-design.md`).

## 1. Motivation

Three connected changes:

1. **Per-mode config blocks.** The two reward constructions (Version A `trait_drives_reward`, Version B `hidden_quality`) have genuinely different parameters. A single flat `PolicyConfig` mixes them and leaves dead params (e.g. `b` is unused in Version B). Splitting into mode-specific blocks makes each config say only what that mode needs.
2. **A `p` parameter for Version B.** Version B currently hardcodes the trait threshold at `m > 0` (a fixed 50/50 base rate). We want to control the expected proportion of traitful prompt-actions, the analogue of `b` in Version A. We expose it as `p` (a probability), converted internally to the threshold `c = Phi^{-1}(1 - p)` per `version_b_notes.md`.
3. **A (gamma x p) sweep.** For Version B, the two most important dials are `gamma` (trait-quality correlation) and `p` (trait base rate). We want to sweep both and view the results as a small-multiples grid.

Naming decision (settled): keep `gamma`/`p` distinct from Version A's `rho`/`b`. They are not the same quantities (different causal roles and mechanisms; see `version_b_notes.md` §3). `alpha` *is* shared because it performs the identical operation in both modes (`q = sigma(alpha * latent)`), so it keeps one name.

## 2. Config layout

`PolicyConfig` holds the structural params shared by both modes plus `mode`; each mode gets its own reward block. The runner reads `PolicyConfig` for `N/K/G/mode`, then the block matching `mode`.

```yaml
PolicyConfig:
  N: 512
  K: 16
  G: 8
  mode: hidden_quality

TraitDrivesRewardConfig:
  rho: 0.3
  b: 0.5
  # alpha omitted -> defaults to 1.0

HiddenQualityConfig:
  gamma: 0.5
  p: 0.5
  # alpha omitted -> defaults to 1.0

TrainConfig:
  steps: 2000
  batch_size: 64
  eta: 0.3
  num_runs: 5
  seed: 290402

OutputConfig:
  results_dir: results
  name: hidden_quality_gamma-0.5
```

- `N/K/G` stay in `PolicyConfig` (shared structure; not duplicated into both reward blocks).
- Only the block matching `mode` needs to be present/correct; the other may be absent.
- The example configs show `alpha` omitted so the default is the visible norm.

## 3. Version-B `p` parameter

In `init_env`, hidden_quality branch, convert the target proportion to a threshold and apply it:

```python
from statistics import NormalDist
...
c = NormalDist().inv_cdf(1 - self._p)
self.s = (m > c).astype(float)
```

- `p = 0.5 -> c = 0`, so existing behaviour is preserved exactly (backward compatible).
- `P(s = 1) = P(m > c) = Phi(-c) = p` by construction, since `m ~ N(0,1)` regardless of `gamma`.
- `statistics.NormalDist` is Python stdlib — no new dependency.
- `p` lives only in `HiddenQualityConfig`; `b` lives only in `TraitDrivesRewardConfig`.

## 4. Defaults and defensive reads

- `Policy.__init__` gains `alpha=1.0` as a default (it is currently a required positional arg). `p=0.5` and `gamma=0.5` already default.
- The runner reads reward params so that both a *missing* key and a *blank* (`None`) value fall back to the default:

```python
alpha = reward_cfg.get("alpha") or 1.0
```

applied to `alpha` in both modes, and to `p` (-> 0.5) and `gamma` (-> 0.5) in Version B. This makes `alpha:` (blank) and an omitted `alpha` behave identically.

## 5. Reuse refactor

Pull the per-config seed loop out of `run_experiment.main()` into a reusable function:

```python
def run_config(policy_cfg, reward_cfg, train_cfg, base_seed, num_runs)
    -> {trait_mean, trait_sem, reward_mean, reward_sem, steps_axis}
```

- Builds the `Policy` with the right kwargs for `mode`, loops `num_runs` seeds (`base_seed + i`), stacks curves, returns mean and SEM.
- `run_experiment.py` calls it once and plots a single run (existing plots unchanged).
- `run_sweep.py` calls it per grid cell.

This keeps the seed-averaging logic in exactly one place.

## 6. The sweep — `run_sweep.py`

A separate entry point so `run_experiment.py` stays "one config, one run":

```
uv run python run_sweep.py --config configs/sweep.yaml
```

`sweep.yaml` is a normal config except `HiddenQualityConfig` holds lists:

```yaml
PolicyConfig:
  N: 512
  K: 16
  G: 8
  mode: hidden_quality

HiddenQualityConfig:
  gamma: [0.3, 0.5, 0.7]
  p:     [0.1, 0.3, 0.5]
  # alpha omitted -> 1.0

TrainConfig:
  steps: 2000
  batch_size: 64
  eta: 0.3
  num_runs: 5
  seed: 290402

OutputConfig:
  results_dir: results
  name: hidden_quality_sweep
```

- The sweep is Version-B specific (gamma x p). `run_sweep.py` asserts `mode == hidden_quality`.
- It runs the Cartesian product of `gamma` (A values) x `p` (B values), calling `run_config` for each cell with the scalar `(gamma, p)` for that cell.

## 7. The grid plot — `plots/grid_trait_reward.pdf` (+ `.png`)

A x B small-multiples, **rows = gamma, columns = p**, every cell sharing one x-axis (step `t`) and one y-axis (level), existing paper style:

- each cell plots `T_t` and `R_t` vs `t` with SEM bands (the existing trait-and-reward content);
- each cell draws a dashed horizontal reference at its own initial `T_0 ~= p` (the column's base rate), so the reader sees the trait *lift above baseline*. Because `T_0 ~= p`, columns start at different heights even under the shared y-axis — expected and informative;
- only the bottom row shows x-axis labels, only the left column shows y-axis labels (minimize redundant ink);
- gamma values label the left edge (one per row), p values label the top edge (one per column);
- one shared legend for the whole figure.
- figure size scales with A and B (roughly `(2.2*B, 1.8*A)` inches).

## 8. Results layout

`results/<sweep_name>/`:
- a copy of `sweep.yaml`,
- `plots/grid_trait_reward.pdf` (+ `.png`),
- `metrics.json` holding the A x B grid of final `T` and `R` (mean +/- SEM) keyed by `(gamma, p)`.

No per-cell subfolders — the single grid is the artifact.

## 9. Decisions made without asking (flagged)

- **rows = gamma, columns = p** (gamma is the primary correlation knob; trivial to flip).
- **per-cell dashed baseline at `p`** — what makes "lift" legible across columns of differing start height.

## 10. Success criteria

- `run_experiment.py` with a restructured single config reproduces current behaviour for both modes (trait/reward curves unchanged for `p = 0.5`, `alpha = 1.0`).
- Omitting or blanking `alpha` yields `alpha = 1.0`; the other mode's block may be absent without error.
- Setting `p` changes the realised trait base rate: measured `mean(s) ~= p` at init.
- `run_sweep.py` over a gamma x p grid produces one `grid_trait_reward.pdf` with A x B cells, shared axes, correct row/column labels, and a `metrics.json` grid of final values.

## 11. Out of scope (YAGNI)

- Sweeping Version-A params (`rho`, `b`) — same machinery would extend, but not built now.
- Sweeping more than two dimensions, or `alpha`.
- Per-cell result subfolders or per-cell standalone plot files.
- Any interaction with the Price-equation check (separate spec).
