**Goal**: empirically test the sampled Price-equation prediction for frozen SAE-defined traits in an LLM. 

## Setup

### Prompt distribution
We need to define a real and finite prompt distribution here. Eg. 
$$
D_{\text{train}} = \text{5000 maths prompts}
$$
$$
D_{\text{feat discovery}} = \text{1000 maths prompts}
$$
$$
D_{\text{eval}} = \text{1000 maths prompts}
$$

Considerations:
- The prompts should be homogeneous enough that reward-feature correlations are stable.
- Tune the difficulty so the model is neither always right nor always wrong. For GRPO, you need reward variation within sampled groups (maybe we could aim for 20-70% success for each group of sampled prompts.)

### Reward model
Use a deterministic verifier. $r(x,a)=1$ if the final answer is correct, otherwise $r(x,a)=0$.

### Which features to track?
We need to find which features to track - which ones correlate with the reward function? For each prompt $x_j$, we can sample some completions from the initial model. 
$$
a_{j,1}, \cdots, a_{j,K} \sim \pi_0(\cdot|x_j)
$$
For each completion:
1. Compute the reward $r_{j,i}$. 
2. Teacher-force the original model with our completed responses. We will then collect SAE activations from this teacher-forcing.
$$
s_{\ell, m}(x_j, a_{j,k}) 
$$
$\ell$ - layer
$m$ - SAE feature index

The SAE feature activation is $f_{\ell,m} (x,a)_\tau$, where $\tau$ indexes completion tokens. 
We can begin by setting the score as 
$$
s_{\ell, m} (x,a) = \max_{\tau \in a } f_{\ell,m} (x,a)_\tau
$$
So each SAE feature index would have its own trait score. 

We need to pick features that satisfy:
- Maybe top 5 positive / negative correlations.
- Active on a large number of samples.

So for each candidate feature, we can calculate 
$$
\rho_{\ell, m} = \text{Corr} (s_{\ell, m}(x,a), r(x,a))
$$
and choose features with a large positive / negative $\rho$. 

Questions:
- What is the point of the teacher forcing here? We don't actually need the logprobs of these, do we? 
- Do we sample these traits from the train set?
- Where do we extract the activations from? Which layer? What about activation strength? Are we ignoring this here or? What does $s$ actually look like? I suppose we have many $s$ models, depending on the feature we are measuring? We can probably plot many graphs here. 
- I think we should just measure the correlation between $s$ and the reward $r$. This would be similar to our older experiments.
- I don't necessarily think that these features need to have any sort of real conceptual meaning tbh- though this would be good. I reckon, if this works, we should do another experiment more related to AI safety. But this should be easy because this would be done in the exact same way. 

## RL training

### Model setup
For now, we want to ignore the transmission term of the Price equation. To do this, we need to ensure that the trait score doesn't change with $t$. This can be done by "freezing" the LLM (ie. we do not do full fine-tuning). To do the RL, we should add LoRA layers to the LLM and train these layers, but the LoRA layers must be added *after* the layer that we extract activations from. 

Eg. if we probe after $L$ layer in the residual stream for activations, we must only add LoRA adapters to layers $\ell \ge L+1$. 

Then, for a fixed completion $a$, 
$$
h^{t+1}_N(x,a) = h^{t}_N(x,a)
$$
and therefore
$$
s^{t+1}(x,a) = s^t(x,a)
$$
### Train the model
We then will just train the model with GRPO + the reward model. 

## Evaluation / measurements

### Price-equation evaluation 
#### Price estimate of $\Delta T$
For each $t$ and $t+1$ we should
1. Sample $x_i \sim D_{\text{eval}}$, $a_{j,i} \sim \pi_t(\cdot|x_i)$. We can take $N$ of such samples, and then increase $N$ to see how the plots change. 
2. For each sampled completion, compute the score $s$.
3. Then calculate 
$$
\omega_{j,i} = \frac{\pi_{t+1}(a_i|x_j)}{\pi_{t}(a_i|x_j)}
$$
4. Compute the Price-estimate
$$
\hat{\Delta T_{\text{Price}}} = \frac{1}{N} \sum^N \omega_{j,i} s_{j,i} -\frac{1}{N}\sum^N s_{j,i}
$$

#### Direct estimate of $\Delta T$
1.  Sample $x_j \sim D_{\text{eval}}$, $a_i \sim \pi_t(\cdot|x_j)$ from the old policy. Calculate an estimate $\hat{T}_t = \frac{1}{N} \sum^N s(a_i, x_j)$.
2. Do this again but from the updated model to get $\hat{T}_{t+1}$.
3. Measure $\Delta \hat{T}_{\text{direct}}$ from the differences in these.  