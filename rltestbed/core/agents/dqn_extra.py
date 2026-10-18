"""DQN extensions: prioritised experience replay and distributional C51.

Both reuse the DQN training loop in :mod:`.dqn` (epsilon schedule, target
network, action masking, checkpoint format) and override only the piece that
makes them different: how transitions are sampled, or what the network predicts.
"""

from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .base import ParamSpec, build_spec
from .dqn import PARAMS, _PRESETS, _BaseDQNAgent

_IMAGE_OK = ["vector", "onehot", "image"]


class _Ring:
    """Fixed-size ring buffer with stable indices (needed for priorities)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: list = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def append(self, item) -> int:
        if len(self.data) < self.capacity:
            self.data.append(item)
            idx = len(self.data) - 1
        else:
            idx = self.pos
            self.data[idx] = item
        self.pos = (idx + 1) % self.capacity
        return idx

    def __getitem__(self, i):
        return self.data[i]


# --------------------------------------------------------------------------- #
class PrioritizedDQNAgent(_BaseDQNAgent):
    SPEC = build_spec(
        name="dqn_per",
        label="DQN + prioritised replay",
        description="DQN that replays surprising transitions more often. Transitions "
        "are sampled in proportion to their last TD error.",
        category="value-based", policy_type="off-policy", obs_types=_IMAGE_OK,
        params=PARAMS + [
            ParamSpec("per_alpha", "Priority exponent α", 0.6, "float",
                      "How strongly priorities skew sampling. 0 = uniform replay.",
                      min=0.0, max=1.0),
            ParamSpec("per_beta", "IS correction β", 0.4, "float",
                      "Importance-sampling correction at the start; anneals to 1.",
                      min=0.0, max=1.0, advanced=True),
        ],
        presets=_PRESETS,
        beginner=(
            "Ordinary DQN replays its memories uniformly at random, so a rare, "
            "important moment (the first time the agent scores) is replayed as often "
            "as a boring one. Prioritised replay remembers how surprised the network "
            "was by each memory and revisits surprising ones more."
        ),
        technical=(
            "Sampling probability P(i) = p_i^α / Σ p^α with p_i = |δ_i| + ε. "
            "The bias this introduces is corrected with importance weights "
            "w_i = (N·P(i))^{-β} / max w, β annealed to 1 over training. New "
            "transitions get the current maximum priority so each is seen at least "
            "once (Schaul et al., 2016). Proportional variant, O(N) sampling."
        ),
        intuition="Study the questions you got wrong, not the ones you already know.",
        update_rule="P(i) ∝ (|δ_i| + ε)^α ;  L = Σ_i w_i · huber(δ_i) ,  w_i = (N P(i))^{−β}",
        pseudocode=(
            "buffer.add(transition, priority=max_priority)\n"
            "batch, idx, P = buffer.sample_proportional(B, alpha)\n"
            "w = (N * P) ** -beta;  w /= w.max()\n"
            "delta = y - Q(s, a)\n"
            "minimise mean(w * huber(delta))\n"
            "buffer.priorities[idx] = |delta| + eps"
        ),
        learns="The same Q-network as DQN.",
        exploration="Linear epsilon-greedy.",
        replay_buffer="Proportional prioritised replay with importance-sampling weights.",
        target_network="Hard sync every `target_update_freq` gradient steps.",
        networks="MLP: obs -> hidden -> ReLU -> hidden -> ReLU -> act_dim Q-values.",
        failure_modes=[
            "Too high α overfits to a few noisy transitions",
            "Without the β correction the learned values are biased",
            "This implementation samples in O(N); very large buffers slow it down",
        ],
        expected_metrics=[
            {"key": "loss", "meaning": "Importance-weighted Huber loss."},
            {"key": "epsilon", "meaning": "Exploration rate."},
            {"key": "priority_max", "meaning": "Largest stored priority; spikes mark surprises."},
            {"key": "per_beta", "meaning": "Current importance-sampling exponent."},
        ],
        recommended_envs=["cartpole", "minatar_breakout", "mountain_car"],
        related=["dqn", "dqn_double", "c51"],
        paper="Schaul et al., 2016. Prioritized Experience Replay. ICLR.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        self.buffer = _Ring(self.params["buffer_size"])
        self.priorities = np.zeros(self.buffer.capacity, dtype=np.float64)
        self.alpha_p = float(self.params.get("per_alpha", 0.6))
        self.beta0 = float(self.params.get("per_beta", 0.4))
        self.beta = self.beta0

    def set_training_progress(self, progress: float):
        super().set_training_progress(progress)
        self.beta = self.beta0 + (1.0 - self.beta0) * float(np.clip(progress, 0, 1))

    def update(self, obs, action, reward, next_obs, done):
        s = self._to_tensor(obs).numpy()
        s2 = self._to_tensor(next_obs).numpy()
        n = len(self.buffer)
        top = self.priorities[:n].max() if n else 1.0
        idx = self.buffer.append((s, int(action), float(reward), s2,
                                   bool(done) and not getattr(self, "last_step_truncated", False)))
        self.priorities[idx] = max(top, 1e-6)
        self.total_steps += 1
        self.epsilon = max(
            self._eps_min,
            float(self.params["epsilon_start"]) - self.total_steps * self._eps_slope,
        )
        if len(self.buffer) < self.batch_size or self.total_steps % 2 != 0:
            return
        self._learn()

    def _learn(self):
        n = len(self.buffer)
        scaled = self.priorities[:n] ** self.alpha_p
        probs = scaled / scaled.sum()
        idx = np.random.choice(n, self.batch_size, replace=False, p=probs)
        weights = (n * probs[idx]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        dev = self.device
        states_t = torch.as_tensor(states, dtype=torch.float32, device=dev)
        next_t = torch.as_tensor(next_states, dtype=torch.float32, device=dev)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=dev)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=dev)
        dones_t = torch.as_tensor(dones.astype(np.float32), device=dev)
        w_t = torch.as_tensor(weights, dtype=torch.float32, device=dev)

        q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.q_net(next_t).argmax(dim=1, keepdim=True)  # double-Q target
            next_q = self.target_net(next_t).gather(1, best).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)
        td = targets - q
        loss = (w_t * nn.functional.smooth_l1_loss(q, targets, reduction="none")).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optim.step()

        self.priorities[idx] = np.abs(td.detach().cpu().numpy()) + 1e-5
        self.updates += 1
        self.losses.append(loss.item())
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def get_metrics(self):
        m = super().get_metrics()
        n = len(self.buffer)
        m["priority_max"] = round(float(self.priorities[:n].max()), 4) if n else 0.0
        m["per_beta"] = round(self.beta, 4)
        return m


# --------------------------------------------------------------------------- #
class _CategoricalNet(nn.Module):
    """Outputs a return distribution per action; forward() returns its mean."""

    def __init__(self, obs_dim, act_dim, hidden, atoms, support):
        super().__init__()
        self.act_dim, self.atoms = act_dim, atoms
        self.register_buffer("support", support)
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim * atoms),
        )

    def dist(self, x):
        logits = self.body(x).view(-1, self.act_dim, self.atoms)
        return torch.softmax(logits, dim=-1)

    def forward(self, x):
        return (self.dist(x) * self.support).sum(-1)


class C51Agent(_BaseDQNAgent):
    SPEC = build_spec(
        name="c51",
        label="C51 (distributional DQN)",
        description="Predicts the whole distribution of returns as 51 probability "
        "bins instead of a single expected value.",
        category="value-based", policy_type="off-policy", obs_types=_IMAGE_OK,
        params=PARAMS + [
            ParamSpec("atoms", "Atoms", 51, "int", "Number of bins in the return distribution.",
                      min=5, max=201, advanced=True),
            ParamSpec("v_min", "Support min", -10.0, "float",
                      "Lowest representable return. Set below the worst episode return.",
                      min=-100000.0, max=100000.0),
            ParamSpec("v_max", "Support max", 10.0, "float",
                      "Highest representable return. Set above the best episode return.",
                      min=-100000.0, max=100000.0),
        ],
        presets=[
            {"name": "cartpole", "label": "CartPole range",
             "description": "Support wide enough for discounted CartPole returns.",
             "overrides": {"v_min": 0.0, "v_max": 100.0}, "episodes": 300},
            {"name": "unit-rewards", "label": "Unit rewards",
             "description": "For tasks whose returns stay within ±10.",
             "overrides": {}, "episodes": 300},
        ],
        beginner=(
            "DQN predicts one number: the average future reward. C51 predicts a "
            "histogram: how likely each possible outcome is. Learning the full shape "
            "turns out to be a richer training signal, even if you only ever act on "
            "the average."
        ),
        technical=(
            "Z(s,a) is categorical over N atoms z_i evenly spaced in [v_min, v_max]. "
            "The target distribution r + γ Z(s′, a*) is projected back onto the "
            "fixed support and the network minimises cross-entropy to it. "
            "a* = argmax_a E[Z(s′,a)] (Bellemare et al., 2017)."
        ),
        intuition="Two actions with the same average can have very different risks; "
        "C51 can see the difference.",
        update_rule="L = − Σ_i (Φ 𝒯Z(s,a))_i · log p_i(s,a) ,  𝒯Z = r + γ Z(s′, argmax_a E Z(s′,a))",
        pseudocode=(
            "p = softmax(net(s))[a]                # (atoms,)\n"
            "a* = argmax_a sum(z * softmax(target(s2)))\n"
            "Tz = clip(r + gamma*(1-done)*z, vmin, vmax)\n"
            "m = project(Tz, softmax(target(s2))[a*]) onto z\n"
            "minimise -sum(m * log p)"
        ),
        learns="A categorical distribution over returns for every action.",
        exploration="Linear epsilon-greedy.",
        replay_buffer="Uniform replay.",
        target_network="Hard sync every `target_update_freq` gradient steps.",
        networks="MLP: obs -> hidden -> hidden -> act_dim × atoms logits.",
        failure_modes=[
            "Returns outside [v_min, v_max] are clipped, silently capping the agent",
            "More parameters and compute per update than DQN",
        ],
        expected_metrics=[
            {"key": "loss", "meaning": "Cross-entropy to the projected target distribution."},
            {"key": "epsilon", "meaning": "Exploration rate."},
            {"key": "q_mean", "meaning": "Mean expected return predicted on the last batch."},
        ],
        recommended_envs=["cartpole", "catch", "minatar_freeway"],
        related=["dqn", "dqn_per"],
        paper="Bellemare, Dabney & Munos, 2017. A Distributional Perspective on RL. ICML.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        p = self.params
        self.atoms = int(p["atoms"])
        self.v_min, self.v_max = float(p["v_min"]), float(p["v_max"])
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")
        dev = torch.device(device)
        support = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.dz = (self.v_max - self.v_min) / (self.atoms - 1)
        make = lambda: _CategoricalNet(obs_dim, act_dim, p["hidden_size"], self.atoms, support)  # noqa: E731
        self.q_net, self.target_net = make().to(dev), make().to(dev)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=p["lr"])
        self.support = support.to(dev)
        self._q_means = deque(maxlen=200)

    def _learn(self):
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        dev = self.device
        s = torch.as_tensor(states, dtype=torch.float32, device=dev)
        s2 = torch.as_tensor(next_states, dtype=torch.float32, device=dev)
        a = torch.as_tensor(actions, dtype=torch.long, device=dev)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=dev).unsqueeze(1)
        d = torch.as_tensor(dones.astype(np.float32), device=dev).unsqueeze(1)
        B = self.batch_size

        with torch.no_grad():
            next_dist = self.target_net.dist(s2)
            a_star = (next_dist * self.support).sum(-1).argmax(1)
            next_p = next_dist[torch.arange(B), a_star]
            tz = (r + self.gamma * (1 - d) * self.support.unsqueeze(0)).clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.dz
            lo, hi = b.floor().long(), b.ceil().long()
            lo[(hi > 0) & (lo == hi)] -= 1
            hi[(lo < self.atoms - 1) & (lo == hi)] += 1
            m = torch.zeros(B, self.atoms, device=dev)
            offset = (torch.arange(B, device=dev) * self.atoms).unsqueeze(1)
            m.view(-1).index_add_(0, (lo + offset).view(-1), (next_p * (hi.float() - b)).view(-1))
            m.view(-1).index_add_(0, (hi + offset).view(-1), (next_p * (b - lo.float())).view(-1))

        dist = self.q_net.dist(s)[torch.arange(B), a]
        loss = -(m * torch.log(dist.clamp_min(1e-8))).sum(-1).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optim.step()
        self.updates += 1
        self.losses.append(loss.item())
        self._q_means.append(float((dist.detach() * self.support).sum(-1).mean()))
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def get_metrics(self):
        m = super().get_metrics()
        m["q_mean"] = round(float(np.mean(self._q_means)), 4) if self._q_means else 0.0
        return m
