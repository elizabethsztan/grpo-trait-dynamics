# Simple Theory

Theoretical plots of average trait level $T$ against GRPO training step $t$.

- **$T_t$** — average presence of a trait in a policy's actions, averaged over all prompts:

$$T_t = \frac{1}{N} \sum_x \sum_a \pi_t(a \mid x)\, s(x,a)$$

where $s(x,a) \in \{0,1\}$ marks whether action $a$ exhibits the trait.

- **$t$** — GRPO training step.

## Setup

A tabular softmax policy over $N$ prompts and $K$ actions. Each prompt–action pair $(x,a)$ has a logit $\ell_t(x,a)$; the policy is

$$\pi_t(a \mid x) = \frac{\exp \ell_t(x,a)}{\sum_{a'} \exp \ell_t(x,a')}, \qquad \ell_0 \equiv 0.$$

**Trait.** Fixed and frozen at init (`Policy.s`, an $N\times K$ table), so $\Delta s_t = 0$. How it relates to reward depends on the `mode`.

### Reward models

The trait must be *correlated* with reward, not equal to it. Two causal stories are available (`Policy.mode`):

**`hidden_quality` (primary) — common cause $z \to r$, $z \to s$.** Reward never depends on the trait. A hidden task-quality score is drawn per pair, $z(x,a) \sim \mathcal N(0,1)$, and drives reward alone:

$$q(x,a) = \sigma(\alpha\, z), \qquad r \sim \text{Bernoulli}(q).$$

The trait is a correlated *side effect* of quality:

$$m(x,a) = \gamma z + \sqrt{1-\gamma^2}\,\xi_{x,a}, \quad \xi \sim \mathcal N(0,1), \qquad s(x,a) = \mathbf 1[m > c].$$

$\gamma$ is the trait–quality coupling; $c = \Phi^{-1}(1-p)$ sets the trait base rate to $p$ (since $m \sim \mathcal N(0,1)$). The trait is per prompt–action, so it varies across pairs.

**`trait_drives_reward` (alternative) — $s \to r$.** The trait is the root cause. It is per-*action* ($\lfloor bK\rfloor$ of $K$ actions traitful, shared across prompts), and reward is generated from it via $u = \rho\,\tilde s + \sqrt{1-\rho^2}\,\epsilon$, $\tilde s = 2s-1$, then $q = \sigma(\alpha u)$.

In both, $\alpha$ is the temperature on the sigmoid which controls how much the latent score (either $z$ or $u$) turns into the reward probability. Eg. how the $q$ values are spread between 0 and 1. 

For small $\alpha$ (eg. 0.1), good and bad actions get similar rewards for the GRPO cannot distinguish them. On the other hand, for large $\alpha$, there would be a near deterministic reward. For our experiments, we can just fix this to $1$. 

See [outline.md](outline.md) for the full theory spec.

### Why `hidden_quality` is the cleaner model

In `trait_drives_reward`, reward is literally generated from the trait, so any rise in $T_t$ is partly baked into the reward design — the trait genuinely carries reward information. `hidden_quality` removes that circularity: reward depends *only* on latent quality $z$, and conditional on quality the trait is uninformative ($s \perp r \mid z$). So if $T_t$ still rises, it is genuine emergent amplification of a merely *spurious* correlate — exactly the phenomenon of interest. It also gives per-pair trait variation and an independent base-rate knob $p$.

## Config

`PolicyConfig.mode` selects the reward model, which picks the matching reward block. Example single-run config:

```yaml
PolicyConfig:
  N: 512                 # prompts
  K: 16                  # actions per prompt
  G: 8                   # actions sampled per prompt per GRPO update
  mode: hidden_quality   # or trait_drives_reward

HiddenQualityConfig:     # used when mode: hidden_quality
  gamma: 0.5             # trait–quality coupling
  p: 0.3                 # trait base rate
  alpha: 1.0             # sigmoid gain in q = σ(α·z)

# TraitDrivesRewardConfig:   # used when mode: trait_drives_reward
#   rho: 0.3                 # trait–reward correlation
#   b: 0.5                   # fraction of the K actions that are traitful
#   alpha: 1.0

TrainConfig:
  steps: 2000
  batch_size: 64   # prompts B per update step
  eta: 0.3         # GRPO learning rate
  num_runs: 5      # repeats; curves averaged, bands are SEM over seeds
  seed: 290402     # base seed; run i uses seed + i

OutputConfig:
  results_dir: results
  name: hidden_quality   # results go to results/<name>/
```

For a **sweep**, make `gamma` and `p` lists instead of scalars (see [configs/config_hidden_qual.yaml](configs/config_hidden_qual.yaml)).

## Algorithm

**Initialise** (`Policy.init_env`) — all logits zero ($\pi_0$ uniform), then build the frozen reward model and trait table for the chosen `mode`:
- `hidden_quality`: draw $z$, set $q = \sigma(\alpha z)$; draw $\xi$, set $s = \mathbf 1[\gamma z + \sqrt{1-\gamma^2}\xi > c]$.
- `trait_drives_reward`: pick $\lfloor bK\rfloor$ traitful actions, set $q = \sigma(\alpha u)$ with $u$ correlated to the trait via $\rho$.

**GRPO step** (`Policy.grpo_step`) — *sample → score → nudge logits*, per update:
1. Sample $B$ prompts; for each, draw $G$ actions from $\pi_t$ (inverse-transform sampling).
2. Draw Bernoulli rewards $r \sim \text{Bernoulli}(q)$ for the sampled actions.
3. Group-normalised advantages $A_{b,i} = (r_{b,i} - \bar r_b)/(\sigma_b + \epsilon)$ — each prompt is its own baseline.
4. Update sampled prompts' logits:

$$\ell_{t+1}(x_b,c) = \ell_t(x_b,c) + \frac{\eta}{G}\sum_{i=1}^{G} A_{b,i}\big[\mathbf 1\{a_{b,i}=c\} - \pi_t(c\mid x_b)\big].$$

Line-by-line walkthrough: [docs/grpo_step.md](docs/grpo_step.md).

## Running

**Single run:**

```bash
uv run python run_experiment.py --config configs/config_hidden_qual.yaml
```

Runs `num_runs` seeds, averages the curves, and writes to `results/<name>/`:

- `config.yaml` — a copy of the config used
- `metrics.json` — per-step `trait_mean/sem` and `reward_mean/sem`
- `plots/expected_trait.{png,pdf}` — $T_t$ vs $t$, with SEM band and the $T_0$ baseline
- `plots/trait_and_reward.{png,pdf}` — $T_t$ and $R_t$ together

**Sweep** (`hidden_quality` only) — grid over `gamma` × `p`:

```bash
uv run python run_sweep.py --config configs/config_hidden_qual.yaml
```

Set `gamma` and `p` as lists in `HiddenQualityConfig`. Each cell is a full `num_runs` run; outputs to `results/<name>/`:

- `metrics.json` — final-step trait/reward (mean + SEM) per `(gamma, p)` cell
- `plots/grid_trait_reward.{png,pdf}` — small-multiples grid, rows = $\gamma$, cols = $p$

**Prediction:** when the trait is positively correlated with quality, both $T_t$ and $R_t$ rise — GRPO optimises reward, and the spuriously-correlated trait is amplified along with it.
