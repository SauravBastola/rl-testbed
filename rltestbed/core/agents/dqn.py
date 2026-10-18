"""Deep Q-Network agents: vanilla DQN and Double DQN.

Both share a replay buffer, target network, linear epsilon-greedy exploration
and action-mask support. Double DQN decouples action *selection* from action
*evaluation* in the target to reduce overestimation bias.
"""

from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .base import Agent, ParamSpec, build_spec


def _q_network(obs_dim: int, act_dim: int, hidden_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, act_dim),
    )


PARAMS = [
    ParamSpec("lr", "Learning rate", 5e-4, "float", "Adam learning rate for the Q-network",
              min=1e-6, max=1.0, recommended=5e-4),
    ParamSpec("gamma", "Discount factor", 0.99, "float",
              "How much future rewards are worth vs immediate ones. 0.9-0.999 typical.",
              min=0.0, max=1.0, recommended=0.99),
    ParamSpec("hidden_size", "Hidden units", 128, "int", "Width of the two hidden layers",
              min=8, max=4096, advanced=True),
    ParamSpec("batch_size", "Batch size", 64, "int", "Transitions sampled per gradient step",
              min=1, max=4096, recommended=64),
    ParamSpec("buffer_size", "Replay buffer size", 50000, "int",
              "Number of recent transitions stored for replay", min=100, advanced=True),
    ParamSpec("target_update_freq", "Target update freq", 200, "int",
              "Gradient steps between target-network syncs", min=1, advanced=True),
    ParamSpec("epsilon_start", "Epsilon start", 1.0, "float",
              "Initial probability of taking a random action"),
    ParamSpec("epsilon_end", "Epsilon end", 0.02, "float",
              "Final exploration rate after decay"),
    ParamSpec("epsilon_decay_steps", "Epsilon decay steps", 10000, "int",
              "Steps over which epsilon decays linearly from start to end"),
    ParamSpec("lr_schedule", "LR schedule", "none", "choice",
              "'linear' anneals the learning rate to 5% of its initial value over training",
              choices=["none", "linear"], advanced=True),
]

_PRESETS = [
    {"name": "quick-demo", "label": "Quick demo",
     "description": "Fast run that visibly learns on easy tasks.",
     "overrides": {"lr": 1e-3}, "episodes": 150},
    {"name": "beginner-default", "label": "Beginner default",
     "description": "Balanced defaults that work on CartPole, Catch and Gridworld.",
     "overrides": {}, "episodes": 400},
    {"name": "stable-baseline", "label": "Stable baseline",
     "description": "Conservative settings for reproducible reference curves.",
     "overrides": {"lr": 3e-4, "batch_size": 128}, "episodes": 800},
    {"name": "fast-iteration", "label": "Fast iteration",
     "description": "Shorter runs to sweep hyperparameters quickly.",
     "overrides": {}, "episodes": 200},
]


def _spec(name: str, label: str, description: str, double: bool) -> Dict[str, Any]:
    extra_docs = (
        {
            "beginner": (
                "DQN learns the *value* of taking each action in each situation "
                "(called a Q-value). It stores past experience in a replay buffer and "
                "learns from small random batches of it, which makes learning stable. "
                "An occasional 'target network' copy keeps the values it learns toward "
                "from shifting too fast."
                if not double else
                "Double DQN is DQN with one fix: when estimating how good the next state "
                "is, it uses the online network to *choose* the best next action but the "
                "target network to *score* it. This reduces a systematic overestimation "
                "bug in vanilla DQN."
            ),
            "technical": (
                "Minimises the smooth-L1 loss between Q(s,a) and the one-step TD target "
                "r + gamma * max_a' Q_target(s',a') (0 if terminal). Gradients are clipped "
                "at norm 10; Adam optimiser; target parameters are hard-synced every "
                "`target_update_freq` updates. Epsilon decays linearly per environment step."
                if not double else
                "Same as DQN but the TD target uses double-Q: a* = argmax_a Q(s',a) "
                "(online net), y = r + gamma * Q_target(s',a*). All other mechanics "
                "(replay, hard target sync, grad clipping, linear epsilon) are identical."
            ),
            "update_rule": (
                r"y = r + γ · max_{a'} Q̄(s′,a′) · (1 − done)" if not double
                else r"a* = argmax_{a'} Q(s′,a′);  y = r + γ · Q̄(s′,a*) · (1 − done)"
            ),
            "pseudocode": (
                "for each episode:\n"
                "    s = env.reset()\n"
                "    while not done:\n"
                "        a = eps-greedy(Q(s))\n"
                "        s2, r, done = env.step(a)\n"
                "        buffer.add(s, a, r, s2, done)\n"
                "        batch = buffer.sample(B)\n"
                "        y = r + gamma * max Q_target(s2) * (1-done)\n"
                "        minimise (Q(s,a) - y)^2   # Huber\n"
                "        every N steps: Q_target <- Q"
                if not double else
                "for each episode:\n"
                "    s = env.reset()\n"
                "    while not done:\n"
                "        a = eps-greedy(Q(s))\n"
                "        s2, r, done = env.step(a)\n"
                "        buffer.add(s, a, r, s2, done)\n"
                "        batch = buffer.sample(B)\n"
                "        a* = argmax_a Q_online(s2, a)\n"
                "        y = r + gamma * Q_target(s2, a*) * (1-done)\n"
                "        minimise huber(Q(s,a), y)\n"
                "        every N steps: Q_target <- Q"
            ),
            "failure_modes": [
                "Overestimation of Q-values (worse in vanilla DQN - use Double)",
                "Learning diverges with too-high lr or too-small replay buffer",
                "Gets stuck if epsilon reaches its floor before the policy is good",
                "Slow start: nothing is learned until the buffer has `batch_size` samples",
            ]
            if not double else
            [
                "Still sensitive to lr; divergence possible with very small buffers",
                "Slightly more compute per update than vanilla DQN (two forward passes)",
            ],
        }
    )
    return build_spec(
        name=name,
        label=label,
        description=description,
        category="value-based",
        policy_type="off-policy",
        status="implemented",
        # pixel observations are flattened for the MLP; check_compat warns about it
        obs_types=["vector", "onehot", "image"],
        params=PARAMS,
        presets=_PRESETS,
        beginner=extra_docs["beginner"],
        technical=extra_docs["technical"],
        intuition=(
            "Q(s,a) answers: 'if I do a here, how much reward can I collect from now on?' "
            "If you know this table for every state, acting greedily on it is optimal. "
            "A neural network generalises the table across similar states, and bootstrapping "
            "(learning each entry partly from other entries) lets knowledge propagate "
            "backwards from rewards to earlier decisions."
        ),
        update_rule=extra_docs["update_rule"],
        pseudocode=extra_docs["pseudocode"],
        lifecycle=(
            "1. Act with epsilon-greedy on the online network.\n"
            "2. Store the transition in the replay buffer.\n"
            "3. Every other step, sample a batch and take one gradient step on the TD error.\n"
            "4. Periodically hard-copy online -> target weights.\n"
            "5. Anneal epsilon linearly across `epsilon_decay_steps` environment steps."
        ),
        learns="A Q-network mapping flattened observations to one value per discrete action.",
        exploration="Linear epsilon-greedy: fully random at first, decaying to `epsilon_end`.",
        replay_buffer="Uniform sampling from the last `buffer_size` transitions.",
        target_network=f"Hard sync every {chr(96)}target_update_freq{chr(96)} gradient steps.",
        networks="MLP: obs -> hidden -> ReLU -> hidden -> ReLU -> act_dim Q-values.",
        failure_modes=extra_docs["failure_modes"],
        expected_metrics=[
            {"key": "loss", "meaning": "Huber loss of TD targets - should trend down and stay low."},
            {"key": "epsilon", "meaning": "Current exploration rate - decays linearly to its floor."},
            {"key": "buffer_size", "meaning": "Transitions stored so far."},
        ],
        related=["dqn_double"] if not double else ["dqn", "ppo"],
        recommended_envs=["cartpole", "catch", "gridworld"] if not double else [],
    )


class _BaseDQNAgent(Agent):
    """Shared machinery for DQN / Double DQN."""

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        p = self.params
        self.obs_dim = obs_dim
        dev = torch.device(device)
        self.q_net = _q_network(obs_dim, act_dim, p["hidden_size"]).to(dev)
        self.target_net = _q_network(obs_dim, act_dim, p["hidden_size"]).to(dev)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=p["lr"])
        self.gamma = p["gamma"]
        self.batch_size = p["batch_size"]
        self.target_update_freq = p["target_update_freq"]
        self.buffer = deque(maxlen=p["buffer_size"])

        self.epsilon = float(p["epsilon_start"])
        eps_start = float(p["epsilon_start"])
        eps_end = max(float(p["epsilon_end"]), 1e-6)
        self._eps_slope = (eps_start - eps_end) / max(float(p["epsilon_decay_steps"]), 1.0)
        self._eps_min = eps_end

        self.total_steps = 0
        self.updates = 0
        self.losses = deque(maxlen=500)
        self._lr_schedule = p.get("lr_schedule", "none")
        self._initial_lr = float(p["lr"])
        self._progress_frac = 0.0

    # ------------------------------------------------------------------ #
    def _to_tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        arr = np.asarray(obs, dtype=np.float32)
        return torch.from_numpy(arr.flatten())

    def get_action(self, obs, valid_actions=None):
        _, mask = self._unpack(obs)
        actions = self._mask_or_range(mask if mask is not None else valid_actions)
        if len(actions) == 1:
            return actions[0]
        state = self._to_tensor(obs).to(self.device)

        if self.TRAIN and np.random.random() < self.epsilon:
            return int(np.random.choice(actions))

        with torch.no_grad():
            q = self.q_net(state.unsqueeze(0)).squeeze(0)
        best = max(actions, key=lambda a: q[a].item())
        return int(best)

    def set_training_progress(self, progress: float):
        self._progress_frac = float(np.clip(progress, 0.0, 1.0))
        self.epsilon = max(
            self._eps_min,
            float(self.params["epsilon_start"]) - (self.total_steps * self._eps_slope),
        )
        if self._lr_schedule == "linear":
            new_lr = self._initial_lr * max(0.05, 1.0 - self._progress_frac)
            for g in self.optim.param_groups:
                g["lr"] = new_lr

    def update(self, obs, action, reward, next_obs, done):
        s = self._to_tensor(obs).numpy()
        s2 = self._to_tensor(next_obs).numpy()
        self.buffer.append((s, int(action), float(reward), s2, bool(done)))
        self.total_steps += 1
        self.epsilon = max(
            self._eps_min,
            float(self.params["epsilon_start"]) - self.total_steps * self._eps_slope,
        )
        if len(self.buffer) < self.batch_size or self.total_steps % 2 != 0:
            return
        self._learn()

    def _next_q(self, next_states_t: torch.Tensor) -> torch.Tensor:
        """TD bootstrap term - overridden by Double DQN."""
        with torch.no_grad():
            return self.target_net(next_states_t).max(dim=1).values

    def _learn(self):
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones.astype(np.float32), device=self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        next_q = self._next_q(next_states_t)
        targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optim.step()

        self.updates += 1
        self.losses.append(loss.item())
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def end_of_episode(self):
        pass

    def get_metrics(self):
        m = {
            "loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "epsilon": round(float(self.epsilon), 4),
            "buffer_size": len(self.buffer),
            "updates": self.updates,
        }
        if self._lr_schedule == "linear":
            m["lr_current"] = float(self.optim.param_groups[0]["lr"])
        return m

    def save_model(self, path):
        torch.save(
            {
                "format": type(self).__name__.lower(),
                "agent_name": self.SPEC["name"],
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optim": self.optim.state_dict(),
                "total_steps": self.total_steps,
                "updates": self.updates,
                "params": self.params,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt.get("target_net", ckpt["q_net"]))
        try:
            self.optim.load_state_dict(ckpt.get("optim"))
        except Exception:
            pass
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates = ckpt.get("updates", 0)


class DQNAgent(_BaseDQNAgent):
    SPEC = _spec(
        "dqn",
        "DQN",
        "Deep Q-Network: value-based off-policy control with experience replay, "
        "a target network and epsilon-greedy exploration.",
        double=False,
    )


class DoubleDQNAgent(_BaseDQNAgent):
    SPEC = _spec(
        "dqn_double",
        "Double DQN",
        "DQN with decoupled action selection/evaluation in the target, reducing "
        "the overestimation bias of vanilla DQN.",
        double=True,
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)

    def _next_q(self, next_states_t):
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            return self.target_net(next_states_t).gather(1, best_actions).squeeze(1)


def _dueling_q_network(obs_dim: int, act_dim: int, hidden_size: int) -> nn.Module:
    class DuelingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(obs_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            )
            self.value_stream = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2, 1))
            self.adv_stream = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2, act_dim))

        def forward(self, x):
            f = self.feature(x)
            v = self.value_stream(f)  # (B,1)
            a = self.adv_stream(f)    # (B,A)
            # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
            return v + a - a.mean(dim=1, keepdim=True)

    return DuelingNet()


class DuelingDQNAgent(_BaseDQNAgent):
    """Dueling DQN — same training loop as DQN, but with dueling architecture."""

    SPEC = build_spec(
        name="dueling_dqn",
        label="Dueling DQN",
        description="DQN with dueling streams: value V(s) + advantage A(s,a) → Q. Better policy evaluation in states where action choice doesn't matter.",
        category="value-based",
        policy_type="off-policy",
        status="implemented",
        # pixel observations are flattened for the MLP; check_compat warns about it
        obs_types=["vector", "onehot", "image"],
        params=PARAMS,
        presets=_PRESETS,
        beginner="Dueling splits the Q-network into two heads after a shared trunk: one estimates how good the *state* is (V), the other how much *each action* is better than average (A). Q = V + (A - mean A). In states where action doesn't matter, the network can just learn V and keep A small.",
        technical="Same loss/target as DQN: y=r+γ max Q_target(s',a'). Architecture: obs→128→ReLU→128→ReLU → [value_stream 64→1, adv_stream 64→|A|] → Q = V + A - mean(A). Dueling helps credit assignment; training loop identical.",
        intuition="Q = V + Advantage: separate 'is this a good place?' from 'is this action better than others here?'",
        update_rule="y = r + γ·max_a' Q̄_dueling(s',a') ; Q = V + A - mean(A)",
        pseudocode="feature = trunk(s)\nV = value_stream(feature)\nA = adv_stream(feature)\nQ = V + A - mean(A)\n# rest as DQN: buffer, sample, Huber loss, target sync",
        lifecycle="Same as DQN: act ε-greedy, buffer, sample batch, one grad step, periodic hard sync, epsilon anneal.",
        learns="Dueling Q-network; same replay/target as DQN.",
        exploration="Linear ε-greedy.",
        replay_buffer="Uniform 50k.",
        target_network="Hard sync every target_update_freq.",
        networks="Dueling MLP: shared 128→128 + dueling heads (V 64→1, A 64→|A|).",
        failure_modes=["Same as DQN, but less overestimation in non-critical states", "Advantage stream can still overfit with tiny buffer"],
        expected_metrics=[{"key": "loss", "meaning": "Huber TD loss"}, {"key": "epsilon", "meaning": "Exploration"}, {"key": "buffer_size", "meaning": "Replay size"}],
        related=["dqn", "dqn_double", "c51"],
        recommended_envs=["cartpole", "catch", "acrobot"],
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        # replace q_net / target_net with dueling version
        dev = torch.device(device)
        self.q_net = _dueling_q_network(obs_dim, act_dim, params["hidden_size"]).to(dev)
        self.target_net = _dueling_q_network(obs_dim, act_dim, params["hidden_size"]).to(dev)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=params["lr"])
