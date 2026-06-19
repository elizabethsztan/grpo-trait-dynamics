import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Policy:

    def __init__ (self, 
                  N, #number of prompts, x
                  K, #number of actions, a
                  G, #how many actions sampled per GRPO update
                  rho, #how much trait and reward are correlated
                  b, #how many of the K actions will show the binary trait? between 0 and 1
                  alpha, #how much latent quality score u maps to reward
                  seed = 290402
                  ):
        

        self._N = N
        self._K = K
        self._G = G
        self._rho = rho
        self._alpha = alpha
        self._b = b

        self._seed = seed

    def init_env(self,
                 ):
    

        self.t = 0
        # at the beginning, all logits are zero
        self.logits = np.zeros((self._N, self._K))

        # decide which of these actions show the trait
        self.s_arr = np.zeros(self._K, dtype=int)
        self.s_arr[:int(self._b*self._K)] = 1

        np.random.seed(self._seed)
        np.random.shuffle(self.s_arr)

        s_tilde = self.s_arr * 2 - 1

        u = self._rho * s_tilde[None, :] + np.sqrt(1 - self._rho ** 2) * np.random.randn(self._N, self._K)
        self.q = sigmoid(self._alpha * u)

    def get_pi(self): #converts logits into probs, which is the policy
        shifted = self.logits - self.logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def get_T(self): #get expected trait level for each prompt
        return np.mean(self.get_pi() @ self.s_arr) #returns shape (N,)

    def expected_reward(self): #get avg reward prob of policy. GRPO increases this
        return np.mean(np.sum(self.get_pi() * self.q, axis=1))

    def grpo_step(self, batch_size, eta, eps=1e-8):
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

