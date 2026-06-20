# Part 1: Theory Prediction Using Tabular Softmax GRPO

## 1.1 Purpose

Tabular softmax policy with finite prompts and finite actions.

The theory section should answer:

> If a fixed trait is correlated with reward, does a GRPO-style update increase the expected trait level $T_t$?

## 1.2 Tabular Policy Setup

Let there be $N$ prompts and $K$ possible actions per prompt.

For each prompt-action pair $(x, a)$, define a logit $\ell_t(x, a)$. The policy is:

$$\pi_t(a \mid x) = \frac{\exp(\ell_t(x, a))}{\sum_{a'} \exp(\ell_t(x, a'))}$$

**Recommended first settings:**

| Parameter | Value |
|-----------|-------|
| N (prompts) | 512 |
| K (actions) | 16 |
| G (sampled actions per prompt) | 8 |
| B (prompts per update) | 64 |
| steps | 1000–3000 |

**Initialise logits as either:**

$$\ell_0(x, a) = 0 \quad \text{or} \quad \ell_0(x, a) \sim \mathcal{N}(0, 0.01^2)$$

## 1.3 Define the Trait

For the theory simulation, use a fixed output-level trait $s(x, a) \in \{0, 1\}$.

For the cleanest first version, make the trait action-level: $s(x, a) = s(a)$.

For example, if $K = 16$, then 8 actions are traitful and 8 are not.

This means the trait is a stable property of the response/action, not a random property of each individual prompt-action pair.

## 1.4 Define Reward Probabilities Correlated with the Trait

Each action has a reward probability $q(x, a) = P(r = 1 \mid x, a)$. The reward is not the trait — instead, reward is statistically correlated with the trait.

There are **two versions** of this we can use. Both produce a trait that is correlated with reward; they differ in the causal story relating the two.

### Version A: Trait drives reward (original)

Here the trait $s$ is the root cause, and reward is generated from it. The causal graph is $s \to r$.

Define:

$$\tilde{s}(a) = 2s(a) - 1 \quad \Rightarrow \quad \tilde{s}(a) \in \{-1, +1\}$$

Then define a latent quality score:

$$u(x, a) = \rho \tilde{s}(a) + \sqrt{1 - \rho^2} \, \epsilon_{x,a}, \quad \epsilon_{x,a} \sim \mathcal{N}(0, 1)$$

Then set:

$$q(x, a) = \sigma(\alpha \, u(x, a))$$

At training time, the sampled reward is:

$$r(x, a) \sim \text{Bernoulli}(q(x, a))$$

The parameter $\rho$ controls the reward–trait correlation.

**Recommended first settings:**

- `rho = 0.3` or `rho = 0.5` for the first positive-correlation run
- Later extensions: `rho = 0` and `rho < 0`

### Version B: Hidden quality drives both reward and trait (common cause)

Here reward does **not** depend on the trait at all. Instead a hidden task-quality variable causes both. The causal graph is $z \to r$ and $z \to s$ (a fork), so the trait is a spurious correlate of reward: conditional on quality, the trait carries no reward information ($s \perp r \mid z$).

For each prompt-action pair, draw a hidden task quality:

$$z(x, a) \sim \mathcal{N}(0, 1)$$

Think of $z$ as "how genuinely good this action is at the task." Reward depends only on task quality:

$$q(x, a) = \sigma(\alpha \, z(x, a)), \qquad r(x, a) \sim \text{Bernoulli}(q(x, a))$$

Define the trait as a correlated side effect of quality:

$$m(x, a) = \gamma z(x, a) + \sqrt{1 - \gamma^2} \, \xi_{x,a}, \quad \xi_{x,a} \sim \mathcal{N}(0, 1)$$

$$s(x, a) = \mathbf{1}[m(x, a) > 0]$$

Here:

- $z$ controls reward.
- $s$ is the trait, now an output-level property $s(x,a)$ (a full $N \times K$ table).
- $\gamma$ controls how correlated the trait is with task quality (and hence with reward).
- Reward never uses $s$ directly.

This is the cleaner causal story: GRPO optimises reward (i.e. $z$), and the trait is amplified only because it co-occurs with quality, not because it is rewarded.

**Notes for Version B:**

- The trait is now per prompt-action, not per action. It is still drawn once and frozen, so $\Delta s_t(x,a) = 0$ and the Price-equation check (§1.7) still holds.
- The threshold $m > 0$ gives a balanced 50/50 trait base rate by symmetry; use $m > c$ for a different base rate.
- The realised reward–trait correlation is mediated by $\gamma$, $\alpha$, and the threshold, so measure it empirically rather than reading it off $\gamma$.

## 1.5 GRPO Simulation Loop

At each training step:

1. Sample a batch of prompts $x_1, \ldots, x_B$.
2. For each prompt $x_b$, sample a group of actions: $a_{b,1}, \ldots, a_{b,G} \sim \pi_t(\cdot \mid x_b)$.
3. Sample verifier rewards: $r_{b,i} \sim \text{Bernoulli}(q(x_b, a_{b,i}))$.
4. Compute group mean reward: $\bar{r}_b = \frac{1}{G} \sum_{i=1}^G r_{b,i}$.
5. Compute group reward standard deviation: $\sigma_b = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_{b,i} - \bar{r}_b)^2}$.
6. Compute GRPO-style advantages: $A_{b,i} = \frac{r_{b,i} - \bar{r}_b}{\sigma_b + \epsilon}$.
   - If $\sigma_b \approx 0$, set all advantages in that group to zero.
7. Update the logits with a policy-gradient-style GRPO step. For sampled prompt $x_b$:

$$\ell_{t+1}(x_b, c) = \ell_t(x_b, c) + \frac{\eta}{G} \sum_{i=1}^G A_{b,i} \left[ \mathbf{1}\{a_{b,i} = c\} - \pi_t(c \mid x_b) \right]$$

Because the GRPO advantages are group-normalised and approximately sum to zero, this often behaves like:

$$\ell_{t+1}(x_b, c) \approx \ell_t(x_b, c) + \frac{\eta}{G} \sum_{i=1}^G A_{b,i} \, \mathbf{1}\{a_{b,i} = c\}$$

Use the full update first, then optionally verify that the simplified one behaves similarly.

**Recommended learning rate:** `eta = 0.03` to `0.1`

## 1.6 Main Theory Quantities

After each update, compute the expected trait level exactly by enumerating all prompts and actions:

$$T_t = \frac{1}{N} \sum_x \sum_a \pi_t(a \mid x) \, s(a)$$

Also compute expected reward:

$$R_t = \frac{1}{N} \sum_x \sum_a \pi_t(a \mid x) \, q(x, a)$$

These are the two main curves.

## 1.7 Theory Plots

### Plot 1: $T_t$ versus GRPO step

- x-axis: $t$
- y-axis: $T_t$
- Run with positive reward–trait correlation, e.g. $\rho = 0.3$ or $\rho = 0.5$

**Prediction:** $T_t$ increases over training. This is the main theoretical prediction.

### Plot 2: Reward and Trait Together

Plot both $R_t$ and $T_t$.

**Prediction for positive $\rho$:**
- Reward increases
- Trait level increases

This shows that GRPO is optimising reward, while the trait is amplified because it is correlated with reward.

### Plot 3: Price-Equation Check

Compute:

$$\omega_t(x, a) = \frac{\pi_{t+1}(a \mid x)}{\pi_t(a \mid x)}$$

Then compute:

$$\text{Cov}_{p_t}(\omega_t, s)$$

where $p_t(x, a) = D(x) \pi_t(a \mid x)$.

Because this is a fixed output-level trait, $\Delta s_t(x, a) = 0$, so we should have:

$$T_{t+1} - T_t = \text{Cov}_{p_t}(\omega_t, s)$$

**Plot:**
- Observed $\Delta T$
- Selection term $\text{Cov}(\omega, s)$

These should match. This plot connects the tabular simulation directly to the theoretical decomposition.