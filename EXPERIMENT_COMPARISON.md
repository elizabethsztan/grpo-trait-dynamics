# Trait-dynamics experiments: mine vs Adil's

Both experiments test the **same effect** — that the Price equation's selection term,
`Cov(ω, z)`, predicts how a trait's population mean drifts under GRPO — but on different
tasks, different observables, and very different sample budgets.

- **Mine:** `llm-activation-experiment/`
- **Adil's:** `adil_src/` (library only in this repo — no top-level driver, so the
  checked-in `DEFAULT_CONFIG` is a *smoke* setting; his real runs use a driver not copied here)

---

## 1. Shared framework

- **Fitness `ω`** in both is the **policy likelihood ratio** across one update, *not* the task reward.
  - Mine: `ω = exp(logp_{t+1} − logp_t)` between adjacent checkpoints — `run_price_eval.py:97`
  - Adil: `ω = exp(post_logprob − pre_logprob)`, pre- vs post-gradient-update — `adil_src/grpo.py:358`
- **Selection covariance** `Cov(ω, z) = E[ωz] − E[ω]·E[z]` — `run_price_eval.py:102`, `adil_src/price.py:15`
- **Frozen trait by construction:** both attach LoRA only to layers *above* the hook layer (12),
  so the trait `z` at layer 12 doesn't move within a step ⇒ the Price **transmission term
  `E[ωΔz]` is structurally ≈ 0** and neither of us estimates it directly.
  - Mine: `src/model.py:43-58`, asserted `max|Δs|<1e-4` each step in `run_grpo.py:13`
  - Adil: `adil_src/lora_freeze.py:33-35`

---

## 2. How the trait is measured — the biggest difference

| | **Mine** | **Adil's** |
|---|---|---|
| Task | GSM8K math (4-shot) | Synthetic multiple-choice arithmetic **with a user "hint" → sycophancy** (`adil_src/data.py:103`) |
| Trait observable | **SAE feature activation** — TopK residual SAE (32K wide, k=100), **max over completion tokens** (`src/trait.py:28`) | **Two traits:** (a) behavioural binary `output_agreement` = answer matches hint (`adil_src/traits.py:45`); (b) `activation_agreement` = **frozen linear diff-of-means probe**, z-scored (`adil_src/activation_probe.py:223`) |
| Feature/direction selection | Phase-1 discovery by Pearson corr with reward → pos/neg/**control** features (`run_feature_discovery.py:54`) | Direction built once from labelled agree/disagree pairs; **no SAE** |
| Hook layer | 12 | 12 (same) |

**In short:** mine tracks *unsupervised SAE dictionary features*; his tracks a *hand-defined
behavioural label + a supervised linear probe*.

---

## 3. How the Cov→drift claim is validated

Same Price identity, `T_t − T_0 = Σ Cov_τ(ω, z) + residual`, but the "observed" side is obtained differently:

| | **Mine** | **Adil's** |
|---|---|---|
| Observed drift source | **dedicated** `direct_samples` fresh draws from π_t *and* π_{t+1} per transition (`run_price_eval.py:88-95`) | eval-bank trait mean recorded each step during training (`adil_src/grpo.py:416`) |
| Granularity | **per-transition** `direct_drift` vs `price` | **cumulative** `T_t − T_0` vs `Σ Cov`, with a **residual** curve (`adil_src/plotting.py:66-72`) |
| Budget | high, isolated from training (8192/side full) | small, reuses the eval loop (4 prompts/dist in smoke) |

His "Observed / Predicted / Residual" plot is the **cumulative** version of the per-transition
`direct_drift` vs `price` agreement I compute. Caveat: a cumulative check can hide per-step
errors that cancel in the running sum; the per-transition check catches them individually.

---

## 4. Estimator subtlety: the ω̄ ≠ 1 bias

- My `price` (`run_price_eval.py:101`) uses `E[ωs] − s̄`, which **assumes ω̄ = 1**. In finite
  samples ω̄ ≈ 1.05, so it differs from the true covariance by `(ω̄ − 1)·s̄`. Since the SAE trait
  is `s ≥ 0` (so `s̄ > 0` on every feature), this is a **systematic positive bias** → the overshoot
  in `grid_price` (orange above blue for positive features, under-magnitude for negatives).
- My `cov` (`run_price_eval.py:102`) subtracts `ω̄·s̄` and does **not** have this bias.
- **Adil does not assume ω̄ = 1.** His raw-covariance reconstruction subtracts `mean(ω)·mean(s)`
  (`adil_src/price.py:15`), and he additionally reports a **self-normalized** estimator
  `Σ(ωs)/Σω − s̄` (`adil_src/price.py:70`) — the Hájek form, which is bias-robust to ω̄≠1 and
  lower-variance. This is the more correct discrete-Price estimand.

**Takeaway for my pipeline:** plot `cov` (not `price`) as the prediction, and/or add a
self-normalized `price_sn` — the overshoot should collapse, especially at the smaller pilot
`price_samples` (64/256) where ω̄ strays furthest from 1.

---

## 5. Training hyperparameters

| | **Mine** | **Adil's** |
|---|---|---|
| Model | **Qwen3.5-2B-Base** (`config.yaml:2`) | **Qwen2.5-0.5B-Instruct** (`adil_src/config.py:18`) |
| Finetune | LoRA r=16, α=32, layers >12 (`src/model.py:53`) | LoRA r=8, α=16, layers >12 (`adil_src/config.py:25`) |
| Learning rate | full **1e-5**; pilot sweep **1e-5 / 3e-5 / 1e-4 / 5e-4** | **5e-5** single (`adil_src/config.py:67`) |
| KL coefficient | **0.01** (k3 estimator vs frozen base) (`config.yaml:39`) | **0.0 — no KL** (`adil_src/config.py:72`) |
| Ratio clip | none (`clip: null`) | GRPO clip 0.2 (`adil_src/config.py:75`) |
| Steps | full 200 / pilot 10 | 5 (smoke; real runs longer — his plot shows 100) |
| Group size G | 8 | 2 (smoke) |
| Prompts/step | full 32 / pilot 16 | 2 (smoke) |
| Optimizer | Adam (`run_grpo.py:31`) | set by the (missing) driver |

Note the different regularisation philosophy: **I bound the policy step with KL (0.01) and no clip;
he uses no KL and a 0.2 ratio clip.** This matters because ω *is* the Price fitness signal.

---

## 6. Sampling budgets for Price / drift

| | **Mine** | **Adil's** |
|---|---|---|
| Price / covariance samples | swept: full **[128, 512, 2048]**, pilot **[64, 256]** (`config.yaml:48`) | **2 per distribution per step** (smoke) × 3 distributions (`adil_src/config.py:87`) |
| Direct-drift samples | full **8192**/side, pilot **128** (`config.yaml:49`) | via eval bank; 4 prompts/dist in smoke |
| ESS diagnostic | yes (`run_price_eval.py:103`) | yes (`adil_src/price.py:32`) |

I draw orders of magnitude more samples and **sweep the sample count** to study estimator
convergence; his checked-in config is a smoke budget (real production budgets are in his driver).

---

## TL;DR

- **Same math** (Price selection covariance; `ω` = policy likelihood ratio; layer-12 hook;
  LoRA-above-hook freeze so no real transmission term).
- **Different observable:** mine = unsupervised SAE features on GSM8K; his = hand-defined
  sycophancy label + supervised linear probe.
- **Different validation:** I resample drift at high budget and check per-transition; he
  accumulates the eval-bank drift and checks `Σ Cov` reconstructs it (residual ≈ 0).
- **Estimator:** my `price` has an `ω̄≠1` overshoot bias; his covariance and self-normalized
  forms don't — I should switch to `cov` / add `price_sn`.
- **Training:** he's 0.5B-Instruct, lr 5e-5, no KL, clip 0.2; I'm 2B-Base, lr 1e-5 (sweeping
  to 5e-4), KL 0.01, no clip.
