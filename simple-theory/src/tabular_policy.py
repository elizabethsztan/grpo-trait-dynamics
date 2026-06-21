import numpy as np
from statistics import NormalDist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Policy:

    def __init__ (self, 
                  N, #number of prompts, x
                  K, #number of actions, a
                  G, #how many actions sampled per GRPO update
                  rho = 0.3, #how much trait and reward are correlated
                  b = 0.5, #how many of the K actions will show the binary trait? between 0 and 1
                  alpha = 1.0, #how much latent quality score u maps to reward
                  mode = "trait_drives_reward", #"trait_drives_reward" (s and r correlated via rho) or "hidden_quality" (z->r, z->s)
                  gamma = 0.5, #version B only: how much trait correlates with hidden quality z
                  p = 0.5,
                  seed = 290402
                  ):


        self._N = N
        self._K = K
        self._G = G
        self._rho = rho
        self._alpha = alpha
        self._b = b
        self._mode = mode
        self._gamma = gamma
        self._p = p

        self._seed = seed

    def init_env(self,
                 ):
        self.t = 0
        self.logits = np.zeros((self._N, self._K))

        np.random.seed(self._seed)

        if self._mode == "trait_drives_reward":
            s_arr = np.zeros(self._K, dtype=int)
            s_arr[:int(self._b * self._K)] = 1
            np.random.shuffle(s_arr)
            s_tilde = s_arr * 2 - 1
            u = self._rho * s_tilde[None, :] + np.sqrt(1 - self._rho ** 2) * np.random.randn(self._N, self._K)
            self.q = sigmoid(self._alpha * u)
            self.s = np.broadcast_to(s_arr, (self._N, self._K)).astype(float)

        elif self._mode == "hidden_quality":
            z = np.random.randn(self._N, self._K)
            self.q = sigmoid(self._alpha * z)
            xi = np.random.randn(self._N, self._K)
            m = self._gamma * z + np.sqrt(1 - self._gamma ** 2) * xi
            if self._p <= 0:
                self.s = np.zeros((self._N, self._K))
            elif self._p >= 1:
                self.s = np.ones((self._N, self._K))
            else:
                c = NormalDist().inv_cdf(1 - self._p)
                self.s = (m > c).astype(float)

        else:
            raise ValueError(f"unknown mode: {self._mode}")

    def get_pi(self): #converts logits into probs, which is the policy
        shifted = self.logits - self.logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def get_T(self): #get expected trait level
        return np.mean(np.sum(self.get_pi() * self.s, axis=1))

    def expected_reward(self): #get avg reward prob of policy. GRPO increases this
        return np.mean(np.sum(self.get_pi() * self.q, axis=1))

    def grpo_step(self, batch_size, eta, eps=1e-8, price_check=False):
        pi = self.get_pi()

        batch_idx = np.random.choice(self._N, size=batch_size, replace=False)
        pi_batch = pi[batch_idx]
        q_batch = self.q[batch_idx]

        cdf = np.cumsum(pi_batch, axis=1)
        draws = np.random.rand(batch_size, self._G)
        actions = (draws[:, :, None] > cdf[:, None, :]).sum(axis=2)

        q_sampled = np.take_along_axis(q_batch, actions, axis=1)
        rewards = (np.random.rand(batch_size, self._G) < q_sampled).astype(float)

        r_mean = rewards.mean(axis=1, keepdims=True)
        r_std = rewards.std(axis=1, keepdims=True)
        advantages = (rewards - r_mean) / (r_std + eps)

        grad = np.zeros((batch_size, self._K))
        rows = np.broadcast_to(np.arange(batch_size)[:, None], (batch_size, self._G))
        np.add.at(grad, (rows, actions), advantages)
        grad -= advantages.sum(axis=1, keepdims=True) * pi_batch
        grad *= eta / self._G

        self.logits[batch_idx] += grad
        self.t += 1

        if not price_check:
            return None

        # Price-equation selection estimator
        # Evalutes on a newly sampled set of actions.
        # Evaluate omega = pi_{t+1}/pi_t and s on a
        updated = self.logits[batch_idx]
        shifted = updated - updated.max(axis=1, keepdims=True)
        exp_updated = np.exp(shifted)
        pi_new_batch = exp_updated / exp_updated.sum(axis=1, keepdims=True)

        rng_state = np.random.get_state()
        eval_draws = np.random.rand(batch_size, self._G)
        eval_actions = (eval_draws[:, :, None] > cdf[:, None, :]).sum(axis=2)
        np.random.set_state(rng_state)

        pi_old_eval = np.take_along_axis(pi_batch, eval_actions, axis=1)
        pi_new_eval = np.take_along_axis(pi_new_batch, eval_actions, axis=1)
        omega = (pi_new_eval / pi_old_eval).ravel()
        s_eval = np.take_along_axis(self.s[batch_idx], eval_actions, axis=1).ravel()

        cov = np.mean(omega * s_eval) - np.mean(omega) * np.mean(s_eval)
        return (batch_size / self._N) * cov

