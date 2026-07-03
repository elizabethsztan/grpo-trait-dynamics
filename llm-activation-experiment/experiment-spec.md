# LLM Activation Experiment — Design Spec

*Part 3 of grpo-trait-dynamics. Date: 2026-07-01.*

## Goal

Empirically test the **sampled Price-equation prediction for frozen SAE-defined traits** in a real LLM. Parts 1–2 (`simple-theory/`) validated, in tabular and neural toy policies, that for a fixed output-level trait $s(x,a)$ the per-step trait change under GRPO is exactly the selection term:

$$T_{t+1} - T_t = \mathrm{Cov}_{p_t}(\omega_t, s), \qquad \omega_t(x,a) = \frac{\pi_{t+1}(a\mid x)}{\pi_t(a\mid x)}.$$

This experiment reproduces that result on **Qwen3.5-2B-Base**, where the trait is a **frozen Qwen-Scope SAE feature** and the policy is trained with GRPO on GSM8K. Because the SAE reads the residual stream at layer $L$ and LoRA is added only to layers $> L$, the trait score is frozen ($\Delta s = 0$), so the Price equation again reduces to pure selection. The core question:

> Does the **sampled** estimator $\widehat{\Delta T}_{\text{Price}} = \frac{1}{N}\sum_i \omega_i s_i - \frac{1}{N}\sum_i s_i$, computed on rollouts sampled *independently of the gradient update*, recover the directly-measured trait drift $\Delta\hat T_{\text{direct}}$ — and how much sample budget $N$ does that recovery need?

See `../simple-theory/project_claim.md` for the full theory, including the bias result (estimating $\omega$ on the gradient rollouts inflates the estimate ~2–3×; independent sampling is a correctness requirement).

## Stack (locked)

| Component | Choice | Notes |
|---|---|---|
| Backbone | **Qwen3.5-2B-Base** | 24 layers, hidden size 2048. Base (not instruct) — matches the SAE training distribution and helps hit the reward-variance window. |
| SAE | **Qwen-Scope** `SAE-Res-Qwen3.5-2B-Base`, layer $L$ | Residual-stream, TopK (k=50/100), width 32K, one SAE per layer 0–23. |
| Probe layer | **$L = 12$** (default), Phase-1 scan over `{10, 12, 16}` | Middle depth: richest/most-interpretable features per SAE literature, ample layers left for LoRA. |
| RL | **Custom minimal GRPO loop** + PEFT-LoRA on layers $> L$ only | Full control over LoRA placement, teacher-forced activation capture, and $\omega$ logprob bookkeeping. |
| Task | **GSM8K** | Deterministic final-answer verifier. |
| Rollouts | **Short CoT via few-shot prompt**, capped tokens | Base model has no chat/thinking template, so a fixed few-shot GSM8K prompt elicits brief reasoning + a parseable `Answer: N` line. Keeps sequence-level $\omega$ variance tractable. |
| Sampling | **temperature 1.0, top-p 1.0, no top-k** (first clean run) | The Price estimator assumes rollouts come from exactly the $\pi_t$ whose logprobs enter $\omega$. Any truncated/tempered sampling makes the sampling distribution $\neq$ the logprob distribution and biases $\omega$. Reward-variance issues are fixed via prompting/dataset difficulty, **not** by sampling tricks. |
| Update | GRPO + **small KL-to-reference term**, **no importance-ratio clipping** at first | KL keeps updates small → lower $\omega$ variance → easier Price recovery. Clipping is deferred (it complicates the update and can shrink signal); add it only if updates are unstable or rollouts are reused across epochs. |
| Reward | $r(x,a) = 1$ if extracted final answer matches gold, else $0$ | |

## Non-goals

- No transmission term. The trait is deliberately frozen; this experiment isolates selection. (Activation-drift / transmission is a later part.)
- No automated test suite (per instruction). Correctness is guarded by in-pipeline assertions and by exact-vs-sampled agreement in the output plots — see [Correctness guards](#correctness-guards).
- No safety-relevant trait semantics yet. Features need not be conceptually meaningful for this validation; a follow-up experiment can reuse the identical pipeline with curated traits.

## Architecture — staged pipeline with cached artifacts

LLM rollouts are expensive, so unlike `simple-theory`'s single unified runner we use **four staged scripts**, each consuming the previous stage's cached output. This allows re-running Price-eval without re-training, swapping datasets, and debugging each stage in isolation.

```
llm-activation-experiment/
  experiment-spec.md            # this file
  pyproject.toml                # adds transformers, peft, datasets, sae deps
  README.md                     # one-line experiment description + run commands (per experimental-principles)
  experiments/
    run_capability_check.py     # Phase 0
    run_feature_discovery.py    # Phase 1
    run_grpo.py                 # Phase 2
    run_price_eval.py           # Phase 3
    configs/
      config.yaml               # full run
      config_smoke.yaml         # tiny smoke config
  src/
    __init__.py
    model.py       # load Qwen base + tokenizer; attach LoRA to layers > L
    sae.py         # load & freeze Qwen-Scope SAE at layer L; encode activations -> features
    trait.py       # teacher-force a completion, read h_L, encode SAE, s = max_tau f_{l,m}
    verifier.py    # GSM8K answer extraction + exact match -> r in {0,1}
    data.py        # GSM8K load + split into D_train / D_feat / D_eval
    grpo.py        # custom GRPO step: group-normalised advantages, LoRA-only update
    logprob.py     # teacher-forced sequence logprob over the exact sampled token span (EOS/truncation-aware; for omega)
  results/         # gitignored: metrics JSONL, features.json, checkpoints, plots
```

All results go under `results/` (gitignored). Config dumped as JSON copy alongside outputs. `uv run python -m ...` only; `LOGGER = logging.getLogger(__name__)` per script.

## Phases

### Phase 0 — Capability check (`run_capability_check.py`)

**Purpose (gate).** Confirm Qwen3.5-2B-Base lands in the **20–70% success** window on GSM8K (GRPO needs within-group reward variance) and pick the working prompt subset / generation settings.

**Steps.** Sample completions from $\pi_0$ on a GSM8K sample (a few hundred prompts) at **temperature 1.0 only** (the clean-run sampling setting — see below); run the verifier; report overall accuracy and the **per-prompt success-rate distribution** (fraction of prompts in the 20–70% band).

**No temperature tuning.** The clean Price run *must* sample at temperature 1.0 so the sampling distribution equals the logprob distribution used in $\omega$; sampling at, say, 0.7 would only be valid if $\omega$ used the temperature-modified policy's logprobs. So Phase 0 does **not** search temperature. If temp-1.0 accuracy sits outside the 20–70% band, we shift difficulty via prompting (`n_shots`), prompt/problem filtering, or a MATH fallback — never via temperature.

**Output.** `results/capability_report.json` — accuracy, success-rate histogram, token cap. **We inspect this before proceeding**; if the model ceilings or floors, adjust difficulty (prompt filtering, `n_shots`, or MATH fallback) before Phase 1.

### Phase 1 — Feature discovery (`run_feature_discovery.py`)

**Purpose.** Find the frozen SAE features to track as traits.

**Steps.** For each prompt $x_j \in D_{\text{feat}}$ (1000):
1. Sample $K$ completions $a_{j,1..K} \sim \pi_0(\cdot\mid x_j)$.
2. Compute reward $r_{j,k}$ via the verifier.
3. **Teacher-force** each `(prompt, completion)` through the base model, read the residual stream at each candidate layer $\ell \in \{10, 12, 16\}$, encode with the frozen Qwen-Scope SAE, and score each feature over completion tokens:
   $$s_{\ell,m}(x,a) = \max_{\tau \in a} f_{\ell,m}(x,a)_\tau.$$
4. For each candidate feature compute $\rho_{\ell,m} = \mathrm{Corr}\big(s_{\ell,m}(x,a),\, r(x,a)\big)$ across all sampled pairs, and its **activity** (fraction of pairs with $s_{\ell,m} > 0$).

**Selection with a held-out validation split.** $D_{\text{feat}}$ is split into a **discovery** and a **validation** half. Scanning 32K SAE features easily produces spurious top correlations, so: rank features by $\rho$ on the discovery half among those **active on at least a minimum fraction** of samples (config `min_activity`, default 0.1), take the **top-5 positive and top-5 negative**, then **confirm each survivor's $\rho$ on the validation half** before tracking it (drop features whose sign flips or whose $|\rho|$ collapses). Pick the winning layer $L$ = the candidate layer whose validated features have the strongest / most-active reward correlations (reported, not hard-coded).

**Control features.** Also select a handful of **random active features with near-zero $\rho$** (passing the same `min_activity` gate). These are tracked through training as negative controls: reward-correlated features should move predictably under GRPO, near-zero-$\rho$ features should not — sharpening the causal-selection story.

**Nuisance diagnostics.** For every candidate/selected feature, log correlations of its score $s$ with completion **length**, **parseability** (did a final answer extract?), and **formatting** proxies, and log reward's correlation with those same nuisances. This flags features that correlate with reward only because correct answers are longer / more parseable, rather than being a genuine trait.

**Teacher-forcing clarification** (answering the plan's open question): teacher-forcing is a single forward pass over the fixed `prompt+completion`. Here it serves one job — read $h_L$ to compute $s$. In Phase 3 the same operation additionally yields the completion's sequence logprob for $\omega$. We do **not** need logprobs in Phase 1.

**Output.** `results/features.json` — chosen layer $L$; the 10 validated feature ids (discovery **and** validation $\rho$/activity); the random control feature ids; per-feature standardisation stats (mean/std of $s$ over the discovery half, for reporting $T$ on a comparable scale); per-feature nuisance correlations (length / parseability / formatting); and the SAE identifier. The 10 validated traits plus the controls are tracked through training.

### Phase 2 — GRPO training (`run_grpo.py`)

**Purpose.** Train the policy with GRPO while keeping the trait frozen.

**Model setup.** Load Qwen3.5-2B-Base; attach **LoRA adapters to layers $> L$ only** (attention + MLP projections). Layers $\le L$, and therefore $h_L$, are frozen. The SAE is frozen throughout.

**GRPO step** (custom, mirrors `simple-theory/src/policy.py` neural step, adapted to sequences):
1. Sample $B$ prompts from $D_{\text{train}}$; draw $G$ completions per prompt from $\pi_t$ (short CoT, capped tokens, no thinking).
2. Reward $r$ from the verifier.
3. Group-normalised advantages $A_{b,i} = (r_{b,i} - \bar r_b)/(\sigma_b + \epsilon)$; zero out degenerate groups ($\sigma_b \approx 0$).
4. Policy-gradient loss $-\frac{1}{BG}\sum A_{b,i}\log\pi_t(a_{b,i}\mid x_b)$ (sequence logprob) **plus a small KL-to-reference penalty** (coefficient $\beta$ = `kl_coef`); **no importance-ratio clipping** in the first version. Adam step on LoRA params only. Sampling for rollouts uses temperature 1.0 / top-p 1.0 / no top-k so the sampling distribution matches the logprobs used downstream for $\omega$.

   **KL definition (unambiguous).** The reference is the **frozen initial policy** $\pi_{\text{ref}} = \pi_0$ — the base model with LoRA at its zero init (identity), so $\pi_{\text{ref}}$ is literally the base backbone. The penalty is the standard GRPO per-token **k3 estimator**, evaluated on the generated completion tokens $\tau$ and averaged over them:
   $$\widehat{\mathrm{KL}}_\tau = \frac{\pi_{\text{ref}}(a_\tau \mid \cdot)}{\pi_t(a_\tau \mid \cdot)} - \log\frac{\pi_{\text{ref}}(a_\tau \mid \cdot)}{\pi_t(a_\tau \mid \cdot)} - 1 \;\ge 0,$$
   added to the loss as $\beta \cdot \frac{1}{BG}\sum_{b,i}\frac{1}{|a_{b,i}|}\sum_\tau \widehat{\mathrm{KL}}_\tau$. ($\pi_{\text{ref}}$ logprobs come from a single frozen forward pass with LoRA disabled.) If a simpler variant is preferred later, the sequence-level sampled penalty $\beta(\log\pi_t(a\mid x) - \log\pi_{\text{ref}}(a\mid x))$ is the drop-in alternative; the k3 token form is the default because it is non-negative and lower-variance.

**Frozen-trait guarantee.** The trait is frozen iff **every LoRA parameter is strictly downstream of the exact activation the SAE reads**. The Qwen-Scope hook point must be confirmed first (see [Pre-implementation verification](#pre-implementation-verification)) — residual-stream SAEs may be trained on the *pre-block*, *post-attention*, or *post-MLP/post-block* residual, and the LoRA boundary is set from that exact point, not from a vague "layer $L$". Given the confirmed hook (call the read activation $h_L^{\text{hook}}$), LoRA is placed only on components downstream of it, so $h_L^{\text{hook},\,t+1} = h_L^{\text{hook},\,t}$ for any fixed completion, hence $s^{t+1} = s^t$, hence $\Delta s = 0$. This is also asserted empirically at runtime (see [Correctness guards](#correctness-guards)).

**Checkpointing.** Save the LoRA adapter state at every step (or every $k$ steps, config `checkpoint_every`) so Phase 3 can reconstruct each $\pi_t$ and $\pi_{t+1}$. Log per-step training metrics (mean reward, mean advantage, degenerate-group fraction) to `results/train_metrics.jsonl`.

**Output.** `results/checkpoints/step_XXXX/` (LoRA adapters), `results/train_metrics.jsonl`.

### Phase 3 — Price-equation evaluation (`run_price_eval.py`)

**Purpose.** For each tracked feature and each step $t$, compare the sampled Price estimate against the direct trait drift.

For each recorded step pair $(t, t+1)$, draw a **fresh** evaluation set from $D_{\text{eval}}$, sampled **independently of the Phase-2 gradient rollouts** (the bias fix from `project_claim.md`):

- **Direct estimate (high budget — the target line).** $\hat T_t = \frac{1}{N_{\text{dir}}}\sum_i s(x_i, a_i)$ with $a_i \sim \pi_t$; independently $\hat T_{t+1}$ with $a_i \sim \pi_{t+1}$. Then $\Delta\hat T_{\text{direct}} = \hat T_{t+1} - \hat T_t$. This uses a **larger** budget $N_{\text{dir}}$ than the Price sweep: the direct drift is itself noisy and serves as the convergence target, so it must be the cleaner line.
- **Price estimate.** Sample $(x_i, a_i) \sim p_t$ (i.e. $x_i \sim D_{\text{eval}}$, $a_i \sim \pi_t(\cdot\mid x_i)$), compute
  $$\omega_i = \exp\big(\log\pi_{t+1}(a_i\mid x_i) - \log\pi_t(a_i\mid x_i)\big)$$
  from teacher-forced sequence logprobs (numerically stable log-space), and
  $$\widehat{\Delta T}_{\text{Price}} = \frac{1}{N}\sum_i \omega_i s_i - \frac{1}{N}\sum_i s_i.$$
  We also report the full covariance form $\overline{\omega s} - \bar\omega\,\bar s$ (does not assume $\bar\omega = 1$).

**Sample-budget sweep.** Recompute the Price estimate at several $N$ (config `price_samples`, a list, all $\le N_{\text{dir}}$) to show convergence to the high-budget direct drift as budget grows — the central practical result.

**$\omega$ diagnostics.** For each step log the importance-ratio distribution — $\bar\omega$, variance, max, and the **effective sample size** $\mathrm{ESS} = (\sum_i \omega_i)^2 / \sum_i \omega_i^2$. Heavy-tailed $\omega$ (low ESS) makes the Price estimate unstable even when the theory holds, and tells us whether a few extreme ratios dominate.

**EOS / truncation handling.** The action $a$ is the exact token sequence sampled during generation. $\omega$ uses the **sequence logprob over precisely those tokens** — including the EOS token when generation stopped on EOS, and over only the emitted tokens (no phantom EOS) when generation hit `max_new_tokens`. The teacher-forced logprob under $\pi_t$ and $\pi_{t+1}$ must be computed over the identical token span; `logprob.py` defines this span once and both policies reuse it.

**Output.** `results/price_eval.jsonl` (per feature, per step, per $N$: direct drift, Price estimate, covariance form, $\bar\omega$, $\omega$ variance/max, ESS) and plots below.

## Outputs & plots

Following `simple-theory` conventions (serif matplotlib style, `.png` + `.pdf`, SEM bands over seeds), per tracked feature:

- `expected_trait` — $T_t$ vs $t$ with SEM band and $T_0$ baseline.
- `price_check` — cumulative **observed** $\sum \Delta\hat T_{\text{direct}}$ vs **predicted** $\sum \widehat{\Delta T}_{\text{Price}}$, with a residual panel (obs − pred).
- `price_convergence` — final-step Price estimate vs $N$, approaching the high-budget direct drift.
- `grid_price` — small-multiples grid across the 10 features (rows = sign of $\rho$), **with control features overlaid/flat** as a reference.
- `omega_diagnostics` — $\bar\omega$ / variance / ESS vs $t$, to read off estimator stability.
- `nuisance_table` — per-feature correlations of $s$ (and of reward) with length / parseability / formatting, so a formatting-proxy "trait" is visible at a glance.

Consult `/plots` and `/tufte-viz` skills when implementing plotting.

## Config

Single `config.yaml` with per-phase blocks, plus a `config_smoke.yaml` tiny variant. Illustrative:

```yaml
ModelConfig:
  backbone: Qwen/Qwen3.5-2B-Base
  sae_repo: Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100
  layer_L: 12                 # probe layer; Phase 1 may override from the scan
  seed: 290402

DataConfig:
  dataset: gsm8k
  n_train: 5000
  n_feat: 1000
  n_eval: 1000

GenConfig:
  temperature: 1.0              # clean run: sampling dist == logprob dist (unbiased omega)
  top_p: 1.0                    # no nucleus truncation
  top_k: 0                      # no top-k truncation
  max_new_tokens: 256
  prompt_style: few_shot        # fixed few-shot GSM8K template; completions end with "Answer: N"
  n_shots: 4

CapabilityConfig:                # Phase 0
  n_prompts: 400
  temperature: 1.0             # clean-run setting; NOT swept (see Phase 0)
  target_success_band: [0.2, 0.7]

FeatureDiscoveryConfig:          # Phase 1
  K: 8                          # completions per prompt
  candidate_layers: [10, 12, 16]
  min_activity: 0.1
  top_k_each_sign: 5
  val_frac: 0.5                 # discovery/validation split of D_feat; selected features re-checked on val
  n_controls: 5                 # random active near-zero-rho control features to track

GRPOConfig:                      # Phase 2
  steps: 200
  batch_size: 32               # prompts B per update
  G: 8                         # completions per prompt
  lr: 1.0e-5
  kl_coef: 0.01                # small KL-to-reference penalty (lowers omega variance)
  clip: null                   # no importance-ratio clipping in the first version
  lora_rank: 16
  lora_layers: ">L"            # layers strictly greater than layer_L
  checkpoint_every: 1
  num_runs: 3                  # seeds; curves averaged, bands = SEM
  seed: 290402

PriceEvalConfig:                 # Phase 3
  price_samples: [128, 512, 2048]   # N sweep for the Price estimator
  direct_samples: 8192              # high-budget N_dir for the target direct-drift line
  eval_prompts: 500

OutputConfig:
  results_dir: results
  name: gsm8k_L12
```

## Pre-implementation verification

One thing must be nailed down **before writing Phase 2/3 code**, because the whole freeze rests on it:

- **Confirm the exact Qwen-Scope hook point.** From the Qwen-Scope model card / repo / loader code, determine precisely which activation the layer-$L$ residual SAE was trained on — pre-block residual, post-attention residual, or post-MLP/post-block residual — and the exact tensor it expects. `src/sae.py` registers a forward hook at *that* point and encodes it; `src/model.py` sets the LoRA target modules to only those strictly downstream. A mismatch (e.g. SAE reads post-MLP of block $L$ but LoRA is added to block $L$'s MLP) silently breaks $\Delta s = 0$. The runtime frozen-trait assertion (guard #1) is the backstop, but the hook convention is verified up front so the assertion is expected to pass, not used to discover the boundary.

## Correctness guards

In lieu of a test suite:

1. **Frozen-trait assertion.** After a GRPO step, teacher-force a held fixed completion under $\pi_t$ and $\pi_{t+1}$; assert $s^{t+1} = s^t$ to machine precision. Fails loudly if any LoRA leaked onto layers $\le L$.
2. **$\bar\omega \approx 1$ sanity + $\omega$/ESS diagnostics.** On independent $\pi_t$ samples $\mathbb E[\omega] = 1$; log $\bar\omega$, $\omega$ variance/max, and ESS each step, and warn on large $\bar\omega$ deviation (signals logprob or sampling bugs) or low ESS (signals a heavy-tailed, unstable estimate).
3. **Verifier spot-check.** Log a handful of `(completion, extracted answer, gold, reward)` rows in Phase 0 for manual inspection.
4. **Exact-vs-sampled agreement.** The Price plots themselves are the end-to-end check: with a frozen trait and independent sampling, $\sum\widehat{\Delta T}_{\text{Price}} \to \sum\Delta\hat T_{\text{direct}}$ as $N$ grows. The Price identity holds for the *actual* $\pi_t \to \pi_{t+1}$ **regardless of the update rule** (KL, clipping, etc. change *which* $\pi_{t+1}$ you reach, not the identity), so a systematic residual points to **estimation noise, a logprob bug, non-independent sampling, or the trait not actually being frozen ($\Delta s \neq 0$)** — not to KL/clipping.
5. **Control features stay flat.** The near-zero-$\rho$ control features should show $\Delta T \approx 0$ throughout; movement in a control is a red flag for a leak in the selection story.

## Scale & compute

- **Smoke first (model-simplification).** `config_smoke.yaml`: ~50 prompts, ~20 GRPO steps, 1 seed, 2–3 features, small token cap — end-to-end on one Swirles GPU in minutes — before the full run.
- **Full run.** 5000 / 1000 / 1000 prompt splits, 200 steps, 3 seeds.
- Swirles GPU (see `/swirles_info`); `uv run python -m ...`; `results/` gitignored.

## Open knobs (sensible defaults set; revisit after smoke run)

- `layer_L` finalized by the Phase-1 scan.
- GRPO `lr`, `lora_rank`, `steps` — tune so the trait drifts visibly within the run.
- SAE width variant (`W32K-L0_50` vs `L0_100`) — start with `L0_100`.

## Prediction

For a feature positively correlated with reward, GRPO amplifies it: both $T_t$ and reward rise, and the sampled selection estimate $\widehat{\Delta T}_{\text{Price}}$ tracks the observed drift $\Delta\hat T_{\text{direct}}$ (to within estimator noise that shrinks with $N$). Negatively-correlated features decline. A persistent residual would indicate effects beyond selection.
