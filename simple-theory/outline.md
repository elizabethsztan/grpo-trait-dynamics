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

# Part 2: Simple Frozen NN with Internal Activation-Level Traits

## 2.1 Purpose

The next step after the tabular theory experiment is to test the same selection story in the smallest possible neural policy, moving from

```text
tabular policy with a fixed trait table
```

to

```text
neural policy with a frozen internal activation direction
```

### Reward model 

An **external reward model** scores the quality of each action the policy produces and rewards the high-quality ones. It sees only the action's quality — never the policy's internal activations. We ask whether an internal **trait** that merely *correlates* with quality is amplified by GRPO even though the reward model never rewards the trait directly. This is exactly the hidden-quality model of Part 1; here we simply read the latent $z$ as the reward model's quality score of the action.

## 2.2 Methodology from Part 1

1. We initialise an $N \times K$ table of prompt-actions.
2. We use the hidden-quality model of trait–reward correlation (Version B).

## 2.3 What is New

Three changes relative to Part 1:

### 1. The trait is a continuous internal activation.

Instead of a binary trait table, the trait is an activation projection:

$$
s(x,a) = v^\top h(x,a),
$$

where $h(x,a) \in \mathbb{R}^d$ is a frozen hidden activation for $(x,a)$ and $v \in \mathbb{R}^d$ is a known frozen trait direction. Only the policy head is trained.

The Part 1 prevalence parameter $p$ becomes

$$
p_{\mathrm{trait}} = P(s(x,a)>0),
$$

the fraction of prompt-actions with nonzero trait activation. This gives both a prevalence knob and a genuinely continuous trait: among active prompt-actions, $s(x,a)$ varies in strength. (The construction and the $p_{\mathrm{trait}}\to\mu_p$ map are in §2.4.)

Because $h$ and $v$ are frozen, $\Delta s_t(x,a)=0$, so this experiment isolates the **selection term** of the Price equation. It does not yet test representation drift or the transmission term.

The main quantity is the expected internal activation:

$$
T_t^{\mathrm{int}}=\mathbb{E}_{x,a\sim\pi_t}[s(x,a)].
$$


### 2. The policy is a NN, but we freeze the NN except the final layer (policy head).

The policy is no longer a free tabular logit table. Instead, logits come from a trainable linear policy head:

$$
\ell_\theta(x,a) = w^\top h(x,a),
$$

and the policy is:

$$
\pi_\theta(a \mid x)
=
\frac{\exp(\ell_\theta(x,a))}{\sum_{a'} \exp(\ell_\theta(x,a'))}.
$$

In the first experiment:

- $h(x,a)$ is generated once and frozen,
- $v$ is generated/defined once and frozen,
- $w$ is trainable,
- GRPO updates only $w$.


### 3. We update the policy via backprop.
Previously, the policy was just a table of logits. But now we update the policy head via backprop.



## 2.4 Setup

Recommended first settings:

| Parameter | Value |
|-----------|-------|
| $N_{\mathrm{train}}$ | `512` |
| $N_{\mathrm{eval}}$ | `512` |
| $K$ actions per prompt | `16` |
| $G$ sampled actions per prompt | `8` |
| $B$ prompts per GRPO update | `64` |
| steps | `1000-3000` |
| hidden dimension $d$ | `16` or `32` |

### Data Generation

For each prompt-action pair $(x,a)$, draw a quality score $z$, a trait preactivation $m$, and the internal trait score $s$. We read $z$ as the score an **external reward model** assigns to the action: it judges the action's quality and rewards the high-quality ones, $r\sim\mathrm{Bernoulli}(q)$, with reward probability depending only on that quality:

$$
q(x,a)=\sigma(\alpha z(x,a)),\qquad z(x,a)\sim\mathcal{N}(0,1).
$$

The trait preactivation is correlated with quality, and the trait is its positive part:

$$
m(x,a)=\mu_p+\gamma z(x,a)+\sqrt{1-\gamma^2}\,\xi_{x,a},\qquad \xi_{x,a}\sim\mathcal{N}(0,1),
$$

$$
s(x,a)=\mathrm{ReLU}(m(x,a))=\max\{0,m(x,a)\}.
$$

Since $m$ is normal with mean $\mu_p$ and unit variance, $P(s>0)=\Phi(\mu_p)$, so we set $\mu_p=\Phi^{-1}(p_{\mathrm{trait}})$ to obtain $P(s>0)=p_{\mathrm{trait}}$. The mean activation is

$$
\mathbb{E}[s]=\phi(\mu_p)+\mu_p\Phi(\mu_p),
$$

so $T_0^{\mathrm{int}}$ depends on both how often the trait is active and how strong the active values are ($\phi,\Phi$ are the standard normal PDF and CDF).

The three knobs are:

- $p_{\mathrm{trait}}$ — sparsity / prevalence of nonzero trait activation,
- $\gamma$ — how strongly the trait activation covaries with reward-relevant quality,
- $\alpha$ — how strongly quality affects reward.

The causal story is a fork: the **reward model sees only the action's quality** $z$ and scores it into a reward; it never reads $m$ or $s$. The trait is a correlated side effect of the same quality, so it is amplified by selection without ever being rewarded:

```text
              quality z(x,a)
             /            \
   [reward model] -> r     m(x,a) -> s(x,a)   (reward model never sees this)
```

Practical $p_{\mathrm{trait}}$ values:

| $p_{\mathrm{trait}}$ | Meaning |
|----------------------|---------|
| `0.05` | very sparse trait activation |
| `0.10` | sparse trait activation |
| `0.30` | moderate trait activation |
| `0.50` | half of prompt-actions have nonzero activation |
| `0.80` | dense trait activation |

Avoid `p_trait = 0.0` or `1.0` as the main setting (edge-case sanity checks only): if no prompt-actions have the trait, there is nothing to amplify.

### Standardisation

Before training, standardise the realised trait scores over the train set:

$$
s \leftarrow \frac{s - \mathbb{E}_{\mathrm{train}}[s]}{\mathrm{Std}_{\mathrm{train}}(s)}.
$$

Do the same for $z$ if useful. This makes effect sizes comparable across seeds and $\gamma$ settings.

## 2.5 Frozen Hidden Activation Construction

The simplest hidden activation is the unrotated raw feature vector $h_0(x,a)$:

$$
h_0(x,a)=
\begin{bmatrix}
z(x,a) \\
s(x,a) \\
\epsilon_1(x,a) \\
\vdots \\
\epsilon_{d-2}(x,a)
\end{bmatrix},
\qquad
\epsilon_j(x,a)\sim\mathcal{N}(0,1).
$$

The unrotated quality and trait directions are:

$$
u_z=[1,0,0,\ldots,0]^\top,
\qquad
v=[0,1,0,\ldots,0]^\top,
$$

so that:

$$
u_z^\top h_0(x,a)=z(x,a),
\qquad
v^\top h_0(x,a)=s(x,a).
$$

In this unrotated case the hidden activation is $h(x,a)=h_0(x,a)$. This is the easiest version to implement and debug.

### Optional Random Rotation

After the unrotated version works, add a fixed random orthogonal rotation so the trait is not simply coordinate 2.

Sample an orthogonal matrix $Q\in\mathbb{R}^{d\times d}$ and define the hidden activation as:

$$
h(x,a)=Q\,h_0(x,a).
$$

The corresponding rotated directions are:

$$
u_z^{\mathrm{rot}}=Qu_z,
$$

and:

$$
v^{\mathrm{rot}}=Qv.
$$

Then:

$$
(u_z^{\mathrm{rot}})^\top h(x,a)=z(x,a),
$$

and:

$$
(v^{\mathrm{rot}})^\top h(x,a)=s(x,a).
$$

Implementation note: if using row-vector conventions, check whether the correct rotated probe is $Qv$ or $Q^\top v$. Add a unit test:

```python
assert torch.allclose(h_rot @ v_rot, s, atol=1e-5)
```

or the equivalent for the chosen convention.

## 2.6 Policy Head

The trainable policy head is linear:

$$
\ell_\theta(x,a)=w^\top h(x,a).
$$

Recommended initialisation:

$$
w_0=0
$$


Using $w_0=0$ gives a uniform initial policy:

$$
\pi_0(a\mid x)=\frac{1}{K}.
$$

Because the same policy head $w$ is shared across all prompts and actions, this is a genuine neural parameterisation rather than a separate tabular logit for every $(x,a)$.

### Trait-orthogonal head (default)

To keep the demonstration airtight, the policy head is **not** allowed to use the trait direction directly. Before training we project $v$ out of the activations the head sees:

$$
h^{\mathrm{pol}}(x,a)=h(x,a)-\big(v^\top h(x,a)\big)\,\hat v,\qquad \hat v=v/\lVert v\rVert,
$$

and form logits from $h^{\mathrm{pol}}$, i.e. $\ell_\theta(x,a)=w^\top h^{\mathrm{pol}}(x,a)$. The trait score $s=v^\top h$ is still measured from the **full** activations, so $T^{\mathrm{int}}$ is unchanged. By construction $w^\top v\equiv 0$: the head can select high-quality actions but has no direct trait shortcut, so any rise in $T^{\mathrm{int}}$ is *pure selection* through the $z$–$s$ correlation rather than a learned trait weight.

The quantity to track is the quality projection:

$$
w_t^\top u_z,
$$
which increases as the head learns to choose higher-quality ($z$) actions. The central claim is the expected internal trait level:

$$
T_t^{\mathrm{int}}=\mathbb{E}_{x,a\sim\pi_t}[v^\top h(x,a)].
$$

Even though the head only ever selects on $z$, $T^{\mathrm{int}}$ rises because high-$z$ actions tend to have larger $s$ when $\gamma>0$ (and falls when $\gamma<0$).

**Diagnostic — the unrestricted head.** If the projection is removed so the head can see $v$ (not the default — it requires deleting the projection step), GRPO transiently *grabs* the trait: early training drives $w^\top v$ up because $s$ is marginally reward-predictive through its correlation with $z$, so $T^{\mathrm{int}}$ overshoots far above the quality-only level. As the head learns that quality $z$ is the true cause, the spurious trait weight is "explained away" ($w^\top v$ decays) and $T^{\mathrm{int}}$ settles back down. This confounder-grab-then-correction is a real effect but a misleading headline plot, which is exactly why the trait-orthogonal head is the default.

## 2.7 GRPO Training Loop

At each training step:

1. Sample a batch of prompts:

$$
x_1,\ldots,x_B.
$$

2. For each prompt, compute logits over all $K$ actions:

$$
\ell_t(x_b,a)=w_t^\top h(x_b,a).
$$

3. Compute the policy:

$$
\pi_t(a\mid x_b)=\mathrm{softmax}_a(\ell_t(x_b,a)).
$$

4. Sample a group of actions:

$$
a_{b,1},\ldots,a_{b,G}\sim\pi_t(\cdot\mid x_b).
$$

5. Sample verifier rewards:

$$
r_{b,i}\sim\mathrm{Bernoulli}(q(x_b,a_{b,i})).
$$

6. Compute group mean reward:

$$
\bar r_b=\frac{1}{G}\sum_{i=1}^G r_{b,i}.
$$

7. Compute group reward standard deviation:

$$
\sigma_b=\sqrt{\frac{1}{G}\sum_{i=1}^G(r_{b,i}-\bar r_b)^2}.
$$

8. Compute group-normalised advantages:

$$
A_{b,i}=\frac{r_{b,i}-\bar r_b}{\sigma_b+\epsilon}.
$$

If $\sigma_b$ is numerically zero, set all advantages in that group to zero.

9. Apply the GRPO policy-gradient loss:

$$
L_{\mathrm{GRPO}}
=-\frac{1}{BG}\sum_{b=1}^B\sum_{i=1}^G
A_{b,i}\log\pi_t(a_{b,i}\mid x_b).
$$

10. Update only $w$ using backprop.

First run with no KL penalty:

$$
\beta_{\mathrm{KL}}=0.
$$

This matches the simplifying assumptions in the theory and Part 1.

After the core result works, optionally add a KL-to-initial-policy penalty:

$$
L=L_{\mathrm{GRPO}}+\beta_{\mathrm{KL}}D_{\mathrm{KL}}(\pi_t\|\pi_0).
$$

The expected effect of KL is to reduce both reward improvement and trait drift.

## 2.8 Exact Evaluation Metrics

Use fixed held-out eval prompts and actions. At each checkpoint, enumerate all eval prompt-action pairs.

The expected reward is:

$$
R_t=
\frac{1}{N_{\mathrm{eval}}}\sum_x\sum_a
\pi_t(a\mid x)q(x,a).
$$

The expected internal trait level is:

$$
T_t^{\mathrm{int}}=
\frac{1}{N_{\mathrm{eval}}}\sum_x\sum_a
\pi_t(a\mid x)v^\top h(x,a).
$$

Since $v^\top h(x,a)=s(x,a)$, this is:

$$
T_t^{\mathrm{int}}=
\frac{1}{N_{\mathrm{eval}}}\sum_x\sum_a
\pi_t(a\mid x)s(x,a).
$$

Track changes relative to initial values:

$$
\Delta T_t=T_t^{\mathrm{int}}-T_0^{\mathrm{int}},
$$

and:

$$
\Delta R_t=R_t-R_0.
$$

Measure the price check:

$$
\Delta T_t \quad \text{vs} \quad \text{Cov}_{p_t}(\omega_t,s)
$$

### Optionally, to track

The expected quality,

$$
Z_t=
\frac{1}{N_{\mathrm{eval}}}\sum_x\sum_a
\pi_t(a\mid x)z(x,a).
$$

The probability mass on active-trait actions,

$$
A_t^{\mathrm{active}}=
\frac{1}{N_{\mathrm{eval}}}\sum_x\sum_a
\pi_t(a\mid x)\mathbf{1}\{s(x,a)>0\}.
$$

and, the change,

$$
\Delta A_t=A_t^{\mathrm{active}}-A_0^{\mathrm{active}},
$$


## 2.9 Exact Price-Equation Check

Because $h$ and $v$ are frozen:

$$
\Delta s_t(x,a)=0.
$$

Therefore, this experiment should satisfy the fixed-trait Price equation exactly:

$$
T_{t+1}^{\mathrm{int}}-T_t^{\mathrm{int}}
=
\mathrm{Cov}_{p_t}(\omega_t,s),
$$

where:

$$
p_t(x,a)=D(x)\pi_t(a\mid x),
$$

and:

$$
\omega_t(x,a)=\frac{\pi_{t+1}(a\mid x)}{\pi_t(a\mid x)}.
$$

Since $D(x)$ is uniform over eval prompts:

$$
D(x)=\frac{1}{N_{\mathrm{eval}}}.
$$

Compute:

$$
\mathrm{Cov}_{p_t}(\omega_t,s)
=
\mathbb{E}_{p_t}[\omega_t s]-\mathbb{E}_{p_t}[\omega_t]\mathbb{E}_{p_t}[s].
$$

Since $\mathbb{E}_{p_t}[\omega_t]=1$, this is:

$$
\mathrm{Cov}_{p_t}(\omega_t,s)
=
\mathbb{E}_{p_t}[\omega_t s]-T_t^{\mathrm{int}}.
$$

But:

$$
\mathbb{E}_{p_t}[\omega_t s]
=
\mathbb{E}_{p_{t+1}}[s]
=T_{t+1}^{\mathrm{int}}.
$$

So the observed and predicted changes should match:

$$
\Delta T_t^{\mathrm{observed}}
=
T_{t+1}^{\mathrm{int}}-T_t^{\mathrm{int}},
$$

$$
\Delta T_t^{\mathrm{Price}}
=
\mathrm{Cov}_{p_t}(\omega_t,s).
$$

Plot both cumulatively.

### Exact vs. sampled estimator

We measure $\mathrm{Cov}_{p_t}(\omega_t,s)$ two ways:

- **Exact (reference).** Enumerate the whole eval table ($N_{\mathrm{eval}}\times K$ pairs, weighted by $\pi_t$). Because everything is enumerated, this equals $\Delta T_t$ to machine precision — a zero-noise check that the decomposition is correct. Only a small toy model makes this possible.
- **Sampled (practical).** On a real model you never enumerate the action space; you only have finite rollouts. So we also estimate the same covariance from `price_samples` fresh draws $(x,a)$ with $x\sim\mathrm{Uniform}(\text{eval})$, $a\sim\pi_t(\cdot\mid x)$ — the unbiased Monte-Carlo estimator. Its error shrinks like $1/\sqrt{\texttt{price\_samples}}$, so sweeping the budget shows how many rollouts are needed to recover the selection term in practice; the exact line is the ground truth it converges to.

The draws must be **fresh** (independent of the actions the GRPO step trained on): reusing the update rollouts biases the estimate upward, because the update raised $\pi_{t+1}$ on exactly those sampled high-advantage actions. (This sampled estimator is the **neural** policy's; the tabular policy, having no eval split, computes only the exact enumerated covariance over its $N\times K$ table.)