# Simple Theory

Theoretical plots of average trait level $T$ against GRPO training step $t$.

- **$T_t$** — average presence of a trait in a policy's actions, averaged over all prompts:

$$T_t = \frac{1}{N} \sum_x \sum_a \pi_t(a \mid x)\, s(a)$$

where $s(a) \in \{0,1\}$ marks whether action $a$ exhibits the trait.

- **$t$** — GRPO training step.

## Setup

A tabular softmax policy over $N$ prompts and $K$ actions. Each prompt–action pair $(x,a)$ has a logit $\ell_t(x,a)$; the policy is

$$\pi_t(a \mid x) = \frac{\exp \ell_t(x,a)}{\sum_{a'} \exp \ell_t(x,a')}, \qquad \ell_0 \equiv 0.$$

**Trait.** Action-level and fixed: $s(a) \in \{0,1\}$, with $\lfloor bK \rfloor$ of the $K$ actions traitful (`Policy.s_arr`). Output-level, so $\Delta s_t = 0$.

**Reward.** Correlated with the trait, not equal to it. With $\tilde s(a) = 2s(a)-1 \in \{-1,+1\}$, a latent quality score

$$u(x,a) = \rho\,\tilde s(a) + \sqrt{1-\rho^2}\;\epsilon_{x,a}, \qquad \epsilon_{x,a} \sim \mathcal N(0,1)$$

gives reward probabilities $q(x,a) = \sigma(\alpha\, u)$ (`Policy.q`), sampled at train time as $r \sim \text{Bernoulli}(q)$. $\rho$ sets the reward–trait correlation ($\rho>0$ for the main run).

See [outline.md](outline.md) for the full theory spec.

## Config

[configs/config.yaml](configs/config.yaml):

```yaml
PolicyConfig:
  N: 512        # prompts
  K: 16         # actions per prompt
  G: 8          # actions sampled per prompt per GRPO update
  rho: 0.3      # trait–reward correlation
  b: 0.5        # fraction of actions that are traitful
  alpha: 1.0    # latent-quality → reward gain in q = σ(α·u)

TrainConfig:
  steps: 2000
  batch_size: 64   # prompts B per update step
  eta: 0.3         # GRPO learning rate
  num_runs: 5      # repeats; curves averaged, bands are SEM over seeds
  seed: 290402     # base seed; run i uses seed + i

OutputConfig:
  output_dir: results
```

## Algorithm

**Initialise** (`Policy.init_env`):
- All logits zero, so $\pi_0$ is uniform over the $K$ actions.
- Assign the trait: $\lfloor bK \rfloor$ actions marked traitful in `s_arr` (shuffled by `seed`).
- Build the reward model $q = \sigma(\alpha u)$ with $u$ correlated to the trait via $\rho$.

**GRPO step** (`Policy.grpo_step`) — *sample → score → nudge logits*, per update:
1. Sample $B$ prompts; for each, draw $G$ actions from $\pi_t$ (inverse-transform sampling).
2. Draw Bernoulli rewards $r \sim \text{Bernoulli}(q)$ for the sampled actions.
3. Group-normalised advantages $A_{b,i} = (r_{b,i} - \bar r_b)/(\sigma_b + \epsilon)$ — each prompt is its own baseline.
4. Update sampled prompts' logits:

$$\ell_{t+1}(x_b,c) = \ell_t(x_b,c) + \frac{\eta}{G}\sum_{i=1}^{G} A_{b,i}\big[\mathbf 1\{a_{b,i}=c\} - \pi_t(c\mid x_b)\big].$$

Line-by-line walkthrough: [docs/grpo_step.md](docs/grpo_step.md).

## Running

```bash
uv run python run_experiment.py --config configs/config.yaml
```

Runs `num_runs` seeds, averages the curves, and writes to `results/` (tagged by `rho`):

- `metrics_rho-<ρ>.json` — config plus per-step `trait_mean/sem` and `reward_mean/sem`
- `trait_rho-<ρ>.{png,pdf}` — $T_t$ vs $t$, with SEM band and the $T_0$ baseline
- `trait_reward_rho-<ρ>.{png,pdf}` — $T_t$ and $R_t$ together

**Prediction:** for $\rho > 0$, both $T_t$ and $R_t$ rise — GRPO optimises reward, and the correlated trait is amplified along with it.
