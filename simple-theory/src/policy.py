import numpy as np
import torch
from statistics import NormalDist
from src.func import sigmoid, relu

class TabularPolicy:

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

        # Exact Price-equation selection term, enumerated over the full N x K table.
        # pi (computed at the top of grpo_step) is pi_t; recompute pi_{t+1} after the update.
        pi_new = self.get_pi()
        omega = pi_new / pi
        p_t = pi / self._N                       # p_t(x,a) = (1/N) pi_t(a|x)
        e_omega_s = np.sum(p_t * omega * self.s)
        e_omega = np.sum(p_t * omega)
        e_s = np.sum(p_t * self.s)
        cov = e_omega_s - e_omega * e_s
        return cov


class NeuralPolicy:

    def __init__ (self,
                  N, #number of prompts, x
                  K, #number of actions, a
                  G, #how many actions sampled per GRPO update
                  d = 32, #hidden activation dimension
                  N_eval = 512, #number of held-out eval prompts
                  b = 0.5, #how many of the K actions will show the binary trait? between 0 and 1
                  alpha = 1.0, #how much latent quality score u maps to reward
                  gamma = 0.5, #how much trait correlates with hidden quality
                  p = 0.5, #proportion of trait-actions with non-zero trait
                  lr = 0.1, #Adam learning rate for the policy head w
                  price_samples = 512, #fresh (x,a) draws for the sampled price-check estimate
                  seed = 290402+1
                  ):


        self._N = N
        self._K = K
        self._G = G
        self._d = d
        self._N_eval = N_eval
        self._alpha = alpha
        self._gamma = gamma
        self._b = b
        self._p = p
        self._lr = lr
        self._price_samples = price_samples

        self._seed = seed

    def _gen_table(self, rng, n): #raw z (quality) and trait score s for an (n, K) table
        z = rng.standard_normal((n, self._K))
        xi = rng.standard_normal((n, self._K))
        noise = self._gamma * z + np.sqrt(1 - self._gamma ** 2) * xi
        if self._p <= 0: #no trait present, nothing to amplify
            s = np.zeros((n, self._K))
        elif self._p >= 1: #always active, ReLU inactive; constant offset cancels in standardisation
            s = noise
        else:
            mu_p = NormalDist().inv_cdf(self._p) #P(s>0) = Phi(mu_p) = p_trait
            s = relu(mu_p + noise) #trait is the positive part of the preactivation
        return z, s

    def _build_h(self, z_std, s_std, rng): #h0 = [z_std, s_std, eps_1, ..., eps_{d-2}], no rotation yet
        n = z_std.shape[0]
        eps = rng.standard_normal((n, self._K, self._d - 2))
        return np.concatenate([z_std[..., None], s_std[..., None], eps], axis=2)

    def init_env(self,
                 ):
        self.t = 0
        self._rng = np.random.default_rng(self._seed)
        self._tgen = torch.Generator().manual_seed(self._seed)
        self._price_gen = torch.Generator().manual_seed(self._seed + 12345)  # price-check draws only

        # frozen data: separate train and held-out eval tables
        z_train, s_train = self._gen_table(self._rng, self._N)
        z_eval, s_eval = self._gen_table(self._rng, self._N_eval)

        # standardise z and s on train stats only (eval uses train mean/std, never its own)
        z_mu, z_sd = z_train.mean(), z_train.std()
        s_mu, s_sd = s_train.mean(), s_train.std()
        z_sd = z_sd or 1.0  # guard against zero-variance (degenerate p) divide-by-zero
        s_sd = s_sd or 1.0
        z_train_std = (z_train - z_mu) / z_sd
        z_eval_std = (z_eval - z_mu) / z_sd
        s_train_std = (s_train - s_mu) / s_sd
        s_eval_std = (s_eval - s_mu) / s_sd

        # frozen activations. by construction u_z = e_1, v = e_2 give u_z.h = z_std, v.h = s_std
        h_train = self._build_h(z_train_std, s_train_std, self._rng)
        h_eval = self._build_h(z_eval_std, s_eval_std, self._rng)

        # reward uses the raw quality z, not the standardised one
        self._h_train = torch.tensor(h_train, dtype=torch.float32)
        self._h_eval = torch.tensor(h_eval, dtype=torch.float32)
        self._q_train = torch.tensor(sigmoid(self._alpha * z_train), dtype=torch.float32)
        self._q_eval = torch.tensor(sigmoid(self._alpha * z_eval), dtype=torch.float32)
        self.s = torch.tensor(s_eval_std, dtype=torch.float32) #trait used in eval metrics

        # trait-orthogonal head: project the trait direction v out of the POLICY features so the
        # head cannot select s directly. self.s is kept from the full activations, so T = E_pi[s]
        # is unchanged; the trait can only rise via the z-s correlation, not a direct shortcut.
        v = torch.zeros(self._d); v[1] = 1.0 #trait probe e_2 (unit); rotation-ready via h @ v
        self._h_train = self._h_train - (self._h_train @ v).unsqueeze(-1) * v
        self._h_eval = self._h_eval - (self._h_eval @ v).unsqueeze(-1) * v

        # trainable policy head, w_0 = 0 -> uniform pi_0 = 1/K
        self._w = torch.zeros(self._d, dtype=torch.float32, requires_grad=True)
        self._optimizer = torch.optim.Adam([self._w], lr=self._lr)

    def get_pi(self, split="eval"): #converts logits (h @ w) into probs, which is the policy
        h = self._h_eval if split == "eval" else self._h_train
        return torch.softmax(h @ self._w, dim=1)

    def get_T(self): #get expected trait level on eval
        with torch.no_grad():
            return float((self.get_pi("eval") * self.s).sum(dim=1).mean())

    def expected_reward(self): #get avg reward prob of policy on eval. GRPO increases this
        with torch.no_grad():
            return float((self.get_pi("eval") * self._q_eval).sum(dim=1).mean())

    def grpo_step(self, batch_size, eta=None, eps=1e-8, price_check=False):
        # eta is accepted-and-ignored so the runner loop is identical to the tabular policy;
        # the neural step uses lr/Adam set in init_env
        if price_check:
            with torch.no_grad():
                pi_t_eval = self.get_pi("eval").clone()

        batch_idx = self._rng.choice(self._N, size=batch_size, replace=False)
        h_batch = self._h_train[batch_idx] #(B, K, d)
        q_batch = self._q_train[batch_idx] #(B, K)

        pi = torch.softmax(h_batch @ self._w, dim=1) #(B, K), with grad

        with torch.no_grad():
            actions = torch.multinomial(pi, self._G, replacement=True, generator=self._tgen) #(B, G)
            q_sampled = torch.gather(q_batch, 1, actions)
            rewards = (torch.rand(batch_size, self._G, generator=self._tgen) < q_sampled).float()
            r_mean = rewards.mean(dim=1, keepdim=True)
            r_std = rewards.std(dim=1, unbiased=False, keepdim=True)
            advantages = (rewards - r_mean) / (r_std + eps)
            advantages = torch.where(r_std < eps, torch.zeros_like(advantages), advantages) #zero degenerate groups

        logp_actions = torch.gather(torch.log(pi + 1e-12), 1, actions) #(B, G)
        loss = -(advantages * logp_actions).mean() #-(1/BG) sum A log pi

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.t += 1

        if not price_check:
            return None

        # Both estimators target the same eval selection term Cov_{p_t}(omega, s),
        # p_t(x,a) = (1/N_eval) pi_t(a|x), omega = pi_{t+1}/pi_t.
        #  - exact: enumerated over the whole eval table (= T_{t+1} - T_t, zero-noise reference).
        #  - sampled: unbiased MC from price_samples fresh (x,a) draws on eval (practical estimate).
        with torch.no_grad():
            pi_tp1_eval = self.get_pi("eval")
            omega = pi_tp1_eval / (pi_t_eval + 1e-12)        # (N_eval, K)

            w_pt = pi_t_eval / self._N_eval
            cov_exact = ((w_pt * omega * self.s).sum()
                         - (w_pt * omega).sum() * (w_pt * self.s).sum())

            # x ~ Uniform(eval prompts), a ~ pi_t(.|x); dedicated RNG so training is unaffected.
            xs = torch.randint(self._N_eval, (self._price_samples,), generator=self._price_gen)
            a_s = torch.multinomial(pi_t_eval[xs], 1, generator=self._price_gen).squeeze(1)
            omega_s = omega[xs, a_s]
            s_s = self.s[xs, a_s]
            cov_sampled = (omega_s * s_s).mean() - omega_s.mean() * s_s.mean()

        return {"exact": float(cov_exact), "sampled": float(cov_sampled)}
