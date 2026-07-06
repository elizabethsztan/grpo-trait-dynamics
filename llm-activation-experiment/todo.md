# TODO — activation-level Price experiment

Status: real 20-step run (`real_lr1e-4`) validates the frozen-trait Price test —
scatter corr=0.99, slope=0.94, controls at origin, ESS 381–473/512. Headline is done.
Below = hardening + extensions, roughly in priority order (cheap+load-bearing first).

## Tier 1 — harden the existing claim (cheap, do first)

- [ ] **Convergence test.** Approximate ω with N = 32 → 512 (→1024?) samples on ONE
      transition, one model. Show cov → direct as N grows (~1/√N spread) and locate where
      low-ESS makes it break. Rig already exists (`run_price_eval.py --max-transitions 1`
      + N sweep; `price_convergence` plot). Cheapest item, hardens the headline most.
- [ ] **High-lr degradation.** Reuse pilot lr=5e-4 (ESS collapse) to show Price degrades
      *predictably* when ω goes heavy-tailed. A boundary-of-validity result: demonstrates we
      know *when* the estimator fails, not only that it works. Near-free (data may exist).
- [ ] **3 seeds.** Re-run the 20-step GRPO ×3 for a CI on slope/corr — turns "slope=0.94"
      into "0.94 ± …". Matters most for the scatter; the grid is already convincing. Medium
      cost. (Possibly only the headline lr needs 3 seeds, not every extension.)

## Tier 2 — new results (medium cost, code changes)

- [ ] **LoRA on all layers → non-stationary trait.** Currently LoRA is strictly downstream
      of the L=12 SAE hook, so s(x,a) is frozen and ΔT = Cov(ω,s) is *exact*. Put LoRA on
      layers ≤ L and the trait's input moves too: s_t and s_{t+1} become different functions
      and ΔT splits into a **policy-shift term (Price/Cov)** + a **representation-shift term**
      E[s_{t+1} − s_t]. Prediction: Price under-predicts, and the residual (direct − cov)
      *measures* representation drift. New result, not just a robustness check. Needs LoRA
      target-module change + care over which s scores the eval.
- [ ] **Off-reward-axis drift.** Formalize the ctrl-22996 observation (ρ≈0 feature made a
      big V-excursion that cov still tracked). Does cov recover drift in ρ≈0 features
      *generally*? Stronger convergent-validity statement than flat controls: Price works off
      the reward axis. Mostly analysis of existing runs.

## Tier 3 — generality across models (highest cost, last)

- [ ] **Second model, activation-level.** Gemma-2-2B has GemmaScope SAEs — rerun the whole
      pipeline to show the Price test isn't Qwen-specific.
- [ ] **Move Adil's output-level experiments over**, unify with the activation-level half;
      run on 1–2 more models if needed.

## Notes
- `trait-drift-parametric.md` documents an ABANDONED idea (predict trajectory shape from the
  reward curve via ρ). Dead because ρ_t is time-dependent and measuring it costs the same
  per-step rollout as computing cov directly (circular). ctrl-22996 is now also a concrete
  counterexample to its amplitude∝ρ prediction. Keep as a record or delete — not on the path.
