# Project Claim

The project claim is that we can measure the per-step change in a trait under GRPO using the covariance between:

- how much the update upweights an output, and
- how much of the trait that output has.

At training step \(t\), define the expected trait level as:

$$
T_t = \mathbb{E}_{x \sim D, a \sim \pi_t(\cdot \mid x)}[s(x,a)].
$$

After one GRPO update, the policy changes from $\pi_t$ to $\pi_{t+1}$. For each sampled prompt-output pair $(x,a)$, define:

$$
\omega_t(x,a) = \frac{\pi_{t+1}(a \mid x)}{\pi_t(a \mid x)}.
$$

This tells us how much the update upweighted or downweighted that output. In neural models, this is measured using log probabilities:

$$
\omega_t(x,a) = \exp(\log \pi_{t+1}(a \mid x) - \log \pi_t(a \mid x)).
$$

Then the selection estimate is:

$$
\widehat{\mathrm{Cov}}_t(\omega, s)
= \overline{\omega s} - \bar{\omega}\bar{s}.
$$

This is computed over outputs sampled from $\pi_t$ that are **independent of the GRPO update** — not the same rollout outputs that produced the gradient. Reusing the gradient outputs biases the estimate upward (see *Estimation Noise* below).

## Fixed Output-Level Traits

For a fixed output-level trait, the trait score \(s(x,a)\) does not change when the model is updated. The same output has the same trait score before and after the update.

In this case, the Price equation reduces to:

$$
T_{t+1} - T_t = \mathrm{Cov}_{p_t}(\omega_t, s).
$$

So the per-step trait change is exactly the selection term. If high-trait outputs are upweighted by GRPO, then:

$$
\mathrm{Cov}(\omega, s) > 0
$$

and the expected trait level increases. If high-trait outputs are downweighted, the covariance is negative and the trait decreases. If the update is unrelated to the trait, the covariance is near zero.

Equivalently:

$$
T_t = T_0 + \sum_{\tau < t} \mathrm{Cov}_{p_\tau}(\omega_\tau, s).
$$

This is the cleanest first claim to validate in the tabular simulation.

## Activation-Level Traits

For activation-level traits, the trait score itself can change because the model's hidden activations can move during training. If:

$$
s_t(x,a) = v_t^\top h_{\theta_t}(x,a),
$$

then both the policy and the representation may change after an update.

The full decomposition is:

$$
T_{t+1} - T_t
= \mathrm{Cov}_{p_t}(\omega_t, s_t)
+ \mathbb{E}_{p_t}[\omega_t \Delta s_t].
$$

The first term is selection: GRPO puts more or less probability mass on already high-trait outputs.

The second term is transmission or representation drift: the internal trait score changes for the same prompt-output pair.

## Project Story

The first goal is to validate that $\mathrm{Cov}(\omega, s)$ measures per-step trait selection for fixed traits.

The next goal is to use the same measurement idea for neural models. The key requirement, established in the tabular validation, is that the selection term must be estimated on outputs sampled **independently of the gradient update**:

1. Do one GRPO update from a rollout sampled at $\pi_t$.
2. Sample a *fresh* evaluation set of outputs from $\pi_t$ (or hold out part of the rollout from the gradient).
3. Measure their trait scores $s(x,a)$.
4. Compute their log probabilities under $\pi_t$ and $\pi_{t+1}$.
5. Compute $\omega = \exp(\log \pi_{t+1} - \log \pi_t)$.
6. Compute $\widehat{\mathrm{Cov}}(\omega, s)$.

Estimating $\omega$ on the same outputs that produced the gradient biases the covariance upward (in the tabular rig, by roughly 2–3×), because those outputs are precisely the ones the update upweighted.

If this covariance predicts the observed trait drift, then selection explains the trait change. If there is a residual, the residual points to representation drift, KL effects, clipping, or other update effects.

The main claim is:

> GRPO amplifies traits that covary with advantage. For fixed traits, the per-step trait change is captured by $\mathrm{Cov}(\omega, s)$. For activation-level traits, observed trait change decomposes into selection plus representation drift.

## Estimation Noise

In the tabular setting, we can compute the exact covariance by enumerating every prompt-action pair. In a neural model, and especially in an LLM, this is impossible: we cannot enumerate all prompts and all possible outputs.

So in practice we estimate the selection term from sampled rollouts:

$$
(x_i, a_i) \sim p_t(x,a) = D(x)\pi_t(a \mid x).
$$

The empirical estimate is:

$$
\widehat{\mathrm{Cov}}_t(\omega, s)
= \frac{1}{M}\sum_i \omega_i s_i
- \left(\frac{1}{M}\sum_i \omega_i\right)
\left(\frac{1}{M}\sum_i s_i\right),
$$

where \(M\) is the number of sampled rollout outputs.

This estimate should approximate the true selection term, but it may be noisy. The main sources of noise are:

- The rollout batch is small compared with the full prompt-output distribution.
- Per-step updates may be small, so most \(\omega_i\) values are close to 1 and the covariance signal is tiny.
- Trait scores \(s_i\) may be noisy if they come from a judge, classifier, benchmark, or linear probe.
- Sequence-level probability ratios can have high variance, especially for long outputs.

This means that a single-step estimate of \(\mathrm{Cov}(\omega, s)\) may be noisy even when the cumulative estimate over many steps tracks trait drift well.

**Estimation bias (distinct from noise).** Beyond variance, there is a *bias*: if the selection term is estimated on the same rollout outputs that produced the gradient, $\widehat{\mathrm{Cov}}(\omega, s)$ is systematically too large. The update raised $\pi$ on exactly those outputs, so they over-represent high-$\omega$ outputs relative to $\pi_t$. In the tabular rig this inflates the per-step estimate by roughly 2–3×. The fix is to estimate $\omega$ and $s$ on outputs sampled independently of the update — a fresh draw from $\pi_t$, or a held-out portion of the rollout. This is a correctness requirement, not a variance-reduction option, and unlike noise it does not average away over steps or seeds.

## Reducing Estimation Noise

There are several ways to make the covariance estimate more stable.

First, average over many GRPO steps. Even if the per-step covariance estimate is noisy, the cumulative prediction:

$$
\sum_{\tau < t} \widehat{\mathrm{Cov}}_\tau(\omega, s)
$$

may track the observed cumulative trait change \(T_t - T_0\) much better than individual step estimates.

Second, average over multiple random seeds. The tabular simulations should report mean curves and uncertainty bands across seeds, because the estimator is stochastic.

Third, increase the evaluation sample size where possible. Larger batch size, larger group size, or more *independent* evaluation rollouts reduce variance by increasing \(M\). These evaluation rollouts must be sampled independently of the gradient update (see the bias note above): independent sampling is required for an unbiased estimate, not merely a faithfulness ablation.

Fourth, use the full covariance formula:

$$
\widehat{\mathrm{Cov}}(\omega, s)
= \overline{\omega s} - \bar{\omega}\bar{s},
$$

rather than assuming \(\bar{\omega} = 1\). In finite samples, \(\bar{\omega}\) will not be exactly 1.

Fifth, compute \(\omega\) from log probabilities:

$$
\omega = \exp(\log \pi_{t+1} - \log \pi_t),
$$

which is numerically more stable than directly dividing probabilities.

Sixth, for long sequence outputs, consider token-normalized or clipped diagnostics as robustness checks. The core Price-equation object is the full sequence ratio, but token-level normalization or mild clipping may reveal whether a small number of extreme ratios dominates the estimate.

The tabular model is useful because it lets us compare three quantities:

1. The true observed trait change \(T_{t+1} - T_t\).
2. The exact full-support \(\mathrm{Cov}(\omega, s)\).
3. The rollout-estimated \(\widehat{\mathrm{Cov}}(\omega, s)\).

This already revealed that the rollout estimate (3) must use outputs sampled independently of the update: reusing the gradient outputs makes (3) overshoot (1) and (2) by roughly 2–3×, while an independent evaluation draw recovers them to within ~1%. With independent sampling, the remaining question is how much rollout data is needed before the sampled covariance is accurate enough to explain trait drift.
