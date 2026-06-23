# Simple Theory

Theoretical plots of average trait level $T$ against GRPO training step $t$.

- **$T_t$** — average presence of a trait in a policy's actions, averaged over all prompts:

$$T_t = \frac{1}{N} \sum_x \sum_a \pi_t(a \mid x)\, s(x,a)$$

where $s(x,a) \in \{0,1\}$ marks whether action $a$ exhibits the trait.

- **$t$** — GRPO training step.

## Setup

A tabular softmax policy over $N$ prompts and $K$ actions. Each prompt–action pair $(x,a)$ has a logit $\ell_t(x,a)$; the policy is

$$\pi_t(a \mid x) = \frac{\exp \ell_t(x,a)}{\sum_{a'} \exp \ell_t(x,a')}, \qquad \ell_0 \equiv 0.$$

**Trait.** Fixed and frozen at init (`TabularPolicy.s`, an $N\times K$ table), so $\Delta s_t = 0$. How it relates to reward depends on the `mode`.

### Reward models

The trait must be *correlated* with reward, not equal to it. Two causal stories are available (`TabularPolicy.mode`):

**`hidden_quality` (primary) — common cause $z \to r$, $z \to s$.** Reward never depends on the trait. Read $z(x,a) \sim \mathcal N(0,1)$ as the quality score an **external reward model** assigns to the action: the reward model sees only the action's quality and rewards the high-quality ones, never inspecting the policy's internal activations. So reward depends on $z$ alone:

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

One shared config file drives both policies; `--mode {tabular,neural}` selects which blocks are read (`TabularPolicyConfig` is shared; the neural-only blocks are read under `--mode neural`). `TabularPolicyConfig.mode` selects the reward model, which picks the matching reward block. Example single-run config ([configs/config.yaml](configs/config.yaml)):

```yaml
TabularPolicyConfig:     # shared: N, K, G, reward mode (read by both --mode values)
  N: 512                 # prompts (= N_train for neural)
  K: 16                  # actions per prompt
  G: 8                   # actions sampled per prompt per GRPO update
  mode: hidden_quality   # or trait_drives_reward (tabular only)

HiddenQualityConfig:     # reward model: q = σ(α·z)
  gamma: 0.5             # trait–quality coupling
  p: 0.3                 # trait base rate
  alpha: 1.0             # sigmoid gain in q = σ(α·z)

NeuralPolicyConfig:      # read only under --mode neural
  N_eval: 512            # held-out eval prompts
  d: 32                  # hidden activation dimension

TrainConfig:             # tabular training
  steps: 2000
  batch_size: 64         # prompts B per update step
  eta: 0.3               # tabular GRPO learning rate
  num_runs: 5            # repeats; curves averaged, bands are SEM over seeds
  seed: 290402           # base seed; run i uses seed + i

NeuralTrainConfig:       # neural training (read under --mode neural)
  steps: 1000
  batch_size: 64
  lr: 0.02               # neural Adam learning rate
  num_runs: 5
  seed: 290402

OutputConfig:
  results_dir: results
  name: hidden_quality   # results go to results/<name>/<mode>/

PriceCheckConfig:
  enabled: true
  samples: 512           # neural sampled-Cov budget per step (neural only)
```

For a **sweep**, make `gamma` and `p` lists instead of scalars (see [configs/config_sweep.yaml](configs/config_sweep.yaml)).

## Algorithm

**Initialise** (`TabularPolicy.init_env`) — all logits zero ($\pi_0$ uniform), then build the frozen reward model and trait table for the chosen `mode`:
- `hidden_quality`: draw $z$, set $q = \sigma(\alpha z)$; draw $\xi$, set $s = \mathbf 1[\gamma z + \sqrt{1-\gamma^2}\xi > c]$.
- `trait_drives_reward`: pick $\lfloor bK\rfloor$ traitful actions, set $q = \sigma(\alpha u)$ with $u$ correlated to the trait via $\rho$.

**GRPO step** (`TabularPolicy.grpo_step`) — *sample → score → nudge logits*, per update:
1. Sample $B$ prompts; for each, draw $G$ actions from $\pi_t$ (inverse-transform sampling).
2. Draw Bernoulli rewards $r \sim \text{Bernoulli}(q)$ for the sampled actions.
3. Group-normalised advantages $A_{b,i} = (r_{b,i} - \bar r_b)/(\sigma_b + \epsilon)$ — each prompt is its own baseline.
4. Update sampled prompts' logits:

$$\ell_{t+1}(x_b,c) = \ell_t(x_b,c) + \frac{\eta}{G}\sum_{i=1}^{G} A_{b,i}\big[\mathbf 1\{a_{b,i}=c\} - \pi_t(c\mid x_b)\big].$$

## Part 2: Neural policy with an internal activation trait

Part 2 moves from the tabular logit table to the smallest *neural* policy: a frozen hidden activation $h(x,a)\in\mathbb{R}^d$ with a single trainable linear head $w$ shared across all prompts, $\ell(x,a)=w^\top h(x,a)$. The trait becomes a continuous internal activation $s(x,a)=v^\top h$ along a frozen probe $v$, so $\Delta s=0$ and the experiment isolates the Price selection term. The same `hidden_quality` reward model applies — an **external reward model** scores only the action's quality $z$ ($q=\sigma(\alpha z)$) and never sees the policy's internal activations $h$.

Differences from Part 1:

- **Trait-orthogonal head (default).** The trait direction $v$ is projected out of the activations the head sees, so the head *cannot* select the trait directly ($w^\top v\equiv 0$). $T^{\mathrm{int}}=\mathbb{E}_\pi[s]$ still rises — purely because high-quality actions tend to be trait-rich when $\gamma>0$ (and falls when $\gamma<0$). The trait is *standardised*, so $T$ is not a probability: it can exceed 1 or go negative.
- **Held-out eval.** A separate frozen eval table ($N_{\mathrm{eval}}$ prompts); all metrics are computed there, since the shared head genuinely generalises.
- **Backprop.** The head is trained with Adam (`lr`), not a manual logit nudge.
- **Price check.** The neural check reports both an **exact** enumerated $\mathrm{Cov}$ (a machine-precision reference, $= \Delta T$) and a **sampled** Monte-Carlo estimate from `price_samples` fresh rollouts — the practical "recover the selection term from finite samples" story, converging to the exact line as `samples` grows. The tabular check is exact only (it has no eval split).

See [outline.md](outline.md) §2 for the full spec.

## Running

Both runners take `--mode {tabular,neural}` (default `tabular`); outputs go to `results/<name>/<mode>/`.

**Single run:**

```bash
uv run python run_experiment.py --config configs/config.yaml --mode tabular
uv run python run_experiment.py --config configs/config.yaml --mode neural
```

Runs `num_runs` seeds, averages the curves, and writes to `results/<name>/<mode>/`:

- `config.yaml` — a copy of the config used
- `metrics.json` — per-step `trait_mean/sem` and `reward_mean/sem`
- `plots/expected_trait.{png,pdf}` — $T_t$ vs $t$, with SEM band and the $T_0$ baseline
- `plots/trait_and_reward.{png,pdf}` — $T_t$ and $R_t$ together
- `plots/price_check_exact.{png,pdf}` — observed $\Delta T$ vs exact $\sum\mathrm{Cov}(\omega,s)$ (both modes)
- `plots/price_check_estimated.{png,pdf}` — observed vs the **sampled** estimator (neural only)

**Sweep** (`hidden_quality` only) — grid over `gamma` × `p`:

```bash
uv run python run_sweep.py --config configs/config_sweep.yaml --mode tabular
uv run python run_sweep.py --config configs/config_sweep.yaml --mode neural
```

Set `gamma` and `p` as lists in `HiddenQualityConfig`. Each cell is a full `num_runs` run; outputs to `results/<name>/<mode>/plots_sweep/`:

- `metrics.json` — final-step trait/reward (mean + SEM) per `(gamma, p)` cell
- `plots_sweep/grid_trait_reward.{png,pdf}` — small-multiples grid, rows = $\gamma$, cols = $p$
- `plots_sweep/grid_price_exact.{png,pdf}` — observed vs exact $\mathrm{Cov}$ per cell (both modes)
- `plots_sweep/grid_price_estimated.{png,pdf}` — observed vs sampled $\mathrm{Cov}$ per cell (neural only)

**Prediction:** when the trait is positively correlated with quality, both $T_t$ and $R_t$ rise — GRPO optimises reward, and the spuriously-correlated trait is amplified along with it.
