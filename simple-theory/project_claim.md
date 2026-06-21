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

This is computed over the sampled outputs from the GRPO rollout batch.

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

The next goal is to use the same measurement idea for neural models:

1. Sample outputs from $\pi_t$.
2. Measure their trait scores $s(x,a)$.
3. Do one GRPO update.
4. Recompute log probabilities under $\pi_{t+1}$) for the same sampled outputs.
5. Compute $\omega = \exp(\log \pi_{t+1} - \log \pi_t)$.
6. Compute $\widehat{\mathrm{Cov}}(\omega, s)$.

If this covariance predicts the observed trait drift, then selection explains the trait change. If there is a residual, the residual points to representation drift, KL effects, clipping, or other update effects.

The main claim is:

> GRPO amplifies traits that covary with advantage. For fixed traits, the per-step trait change is captured by $\mathrm{Cov}(\omega, s)$. For activation-level traits, observed trait change decomposes into selection plus representation drift.
