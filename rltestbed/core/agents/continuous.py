"""Continuous-control agents: DDPG, TD3 and SAC.

These three algorithms unlock the MuJoCo locomotion/manipulation suite and the
continuous classic-control environments, all of which have ``Box`` action
spaces that the discrete agents cannot express.

Shared design
-------------
All three are off-policy actor-critic methods built on the same machinery:

* A **deterministic or Gaussian actor** that emits actions in a normalised
  ``[-1, 1]`` range, rescaled to the environment's real action limits via
  :meth:`Agent.set_action_bounds`.
* **Target networks** updated by Polyak averaging (``tau``) rather than the
  periodic hard sync used by DQN - soft updates are markedly more stable for
  continuous control.
* A **numpy-backed replay buffer** with pre-allocated arrays, which matters at
  MuJoCo step counts (hundreds of thousands of transitions).
* Optional **running observation normalisation**. MuJoCo observations mix
  positions, angles and velocities with wildly different scales (Humanoid even
  includes unbounded contact forces), so feeding raw values to an MLP makes
  learning brittle. This is on by default.

They differ in how they handle the two failure modes of naive DDPG:
overestimation bias and brittle hyperparameter sensitivity.

- **DDPG**  single critic + deterministic policy; exploration by adding noise
  to the action. Simple, but notoriously sensitive and prone to overestimating
  Q-values.
- **TD3**   twin critics (take the min), delayed policy updates, and target
  policy smoothing. Three targeted fixes on top of DDPG.
- **SAC**   stochastic policy trained to maximise *entropy-regularised*
  return, with the temperature tuned automatically. Usually the most robust
  and the best default for a new continuous task.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent, ParamSpec, build_spec

# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #


class ReplayBuffer:
    """Fixed-size ring buffer with pre-allocated numpy arrays.

    Actions are stored in the *normalised* ``[-1, 1]`` range so the critics
    always see bounded inputs, regardless of the environment's real limits.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.max_size = int(size)
        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------- #
# Observation normalisation
# --------------------------------------------------------------------------- #


class RunningNormalizer:
    """Online mean/standard-deviation normalisation (Welford's algorithm)."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        self.dim = int(dim)
        self.clip = float(clip)
        self.eps = float(eps)
        self._mean = np.zeros(self.dim, dtype=np.float64)
        self._m2 = np.zeros(self.dim, dtype=np.float64)
        self.count = 0

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (x - self._mean)

    def normalize(self, x) -> np.ndarray:
        """Normalise one observation or a batch (shape is preserved)."""
        x = np.asarray(x, dtype=np.float64)
        if self.count < 2:
            return x.astype(np.float32)
        std = np.sqrt(self._m2 / self.count) + self.eps
        return np.clip((x - self._mean) / std, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "count": self.count,
            "clip": self.clip,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self._mean = np.asarray(state["mean"], dtype=np.float64)
        self._m2 = np.asarray(state["m2"], dtype=np.float64)
        self.count = int(state.get("count", 0))
        self.clip = float(state.get("clip", self.clip))


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #


def _mlp(sizes: List[int], output_activation=None) -> nn.Module:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class SimBaBlock(nn.Module):
    """Pre-norm residual block with a 4x inverted bottleneck (SimBa, Lee et al. 2025)."""

    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


def _trunk(in_dim: int, hidden: int, n_hidden: int, network: str = "mlp") -> nn.Module:
    """The shared body of actors and critics.

    ``mlp``: Linear+ReLU layers (the classic choice). ``simba``: a linear
    embedding, residual pre-norm blocks and a final LayerNorm. SimBa's argument
    is that this simplicity bias lets networks grow without overfitting, so
    scaling parameters improves sample efficiency (observation normalisation,
    its third part, is the agents' normalize_obs).
    """
    if network == "simba":
        return nn.Sequential(nn.Linear(in_dim, hidden),
                             *[SimBaBlock(hidden) for _ in range(max(1, n_hidden - 1))],
                             nn.LayerNorm(hidden))
    return _mlp([in_dim] + [hidden] * n_hidden, output_activation=nn.ReLU)


class DeterministicActor(nn.Module):
    """Maps observations to a deterministic action in ``[-1, 1]`` (tanh)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256,
                 n_hidden: int = 2, network: str = "mlp"):
        super().__init__()
        if network == "mlp":  # unchanged layout, so old checkpoints still load
            self.net = _mlp([obs_dim] + [hidden_size] * n_hidden + [act_dim],
                            output_activation=nn.Tanh)
        else:
            self.net = nn.Sequential(_trunk(obs_dim, hidden_size, n_hidden, network),
                                     nn.Linear(hidden_size, act_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QCritic(nn.Module):
    """Q(s, a) with the action concatenated to the observation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256,
                 n_hidden: int = 2, network: str = "mlp"):
        super().__init__()
        if network == "mlp":
            self.net = _mlp([obs_dim + act_dim] + [hidden_size] * n_hidden + [1])
        else:
            self.net = nn.Sequential(_trunk(obs_dim + act_dim, hidden_size, n_hidden, network),
                                     nn.Linear(hidden_size, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class TwinQCritic(nn.Module):
    """Two independent critics; TD3/SAC take the min to curb overestimation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256,
                 n_hidden: int = 2, network: str = "mlp"):
        super().__init__()
        self.q1 = QCritic(obs_dim, act_dim, hidden_size, n_hidden, network)
        self.q2 = QCritic(obs_dim, act_dim, hidden_size, n_hidden, network)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy with the reparameterisation trick.

    Sampling is ``a = tanh(mu + std * eps)`` with ``eps ~ N(0, I)``, which keeps
    the path differentiable while still injecting exploration noise.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256,
                 n_hidden: int = 2, network: str = "mlp"):
        super().__init__()
        if network == "mlp":  # original layout: _mlp leaves the last layer linear
            self.trunk = _mlp([obs_dim] + [hidden_size] * n_hidden)
        else:
            self.trunk = _trunk(obs_dim, hidden_size, n_hidden, network)
        last = hidden_size
        self.mu = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std.exp()

    def sample(self, obs, eps: float = 1e-6):
        mu, std = self.forward(obs)
        normal = torch.randn_like(mu)
        pre_tanh = mu + std * normal
        action = torch.tanh(pre_tanh)
        # Change of variables for the tanh squashing.
        log_prob = (
            -0.5 * (normal ** 2) - torch.log(std + eps) - math_log(2.0 * np.pi) / 2.0
        ).sum(dim=-1, keepdim=True)
        log_prob -= torch.log((1.0 - action ** 2).clamp(min=eps) + eps).sum(
            dim=-1, keepdim=True
        )
        return action, log_prob

    def deterministic(self, obs):
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


def math_log(x: float) -> float:
    """Small helper so the module has no top-level `math` import."""
    return float(np.log(x))


# --------------------------------------------------------------------------- #
# Shared base
# --------------------------------------------------------------------------- #

_COMMON_PARAMS = [
    ParamSpec("gamma", "Discount factor", 0.99, "float",
              "How much future rewards are worth vs immediate ones.",
              min=0.0, max=1.0, recommended=0.99),
    ParamSpec("tau", "Soft update rate", 0.005, "float",
              "Polyak averaging coefficient for the target networks. "
              "Smaller = slower, more stable targets.",
              min=1e-5, max=1.0, advanced=True),
    ParamSpec("batch_size", "Batch size", 256, "int",
              "Transitions sampled per gradient step", min=1, max=4096,
              recommended=256),
    ParamSpec("buffer_size", "Replay buffer size", 100000, "int",
              "Stored transitions. 100k keeps memory sane even for Humanoid's "
              "376-dim observations; raise it if you have RAM to spare.",
              min=100, advanced=True),
    ParamSpec("hidden_size", "Hidden units", 256, "int",
              "Width of each hidden layer in actor and critic",
              min=8, max=4096, advanced=True),
    ParamSpec("lr", "Actor learning rate", 3e-4, "float",
              "Adam learning rate for the policy network",
              min=1e-6, max=1.0, recommended=3e-4),
    ParamSpec("critic_lr", "Critic learning rate", 3e-4, "float",
              "Adam learning rate for the Q-network(s)",
              min=1e-6, max=1.0, recommended=3e-4),
    ParamSpec("warmup_steps", "Warmup steps", 1000, "int",
              "Random actions collected before learning starts, so the buffer "
              "is not filled by one early, bad policy.",
              min=0, max=1000000, advanced=True),
    ParamSpec("updates_per_step", "Updates per env step", 1, "int",
              "Gradient steps taken per environment step. Raise for very short "
              "episodes.", min=1, max=64, advanced=True),
    ParamSpec("normalize_obs", "Normalize observations", True, "bool",
              "Whiten observations with running mean/std. Strongly recommended "
              "for MuJoCo, where scales differ wildly across dimensions.",
              advanced=True),
]


class _ContinuousBase(Agent):
    """Shared plumbing for the off-policy continuous-control agents."""

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any],
                 device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.obs_dim = int(obs_dim)
        self.dev = torch.device(device)

        p = self.params
        self.gamma = float(p["gamma"])
        self.tau = float(p["tau"])
        self.batch_size = int(p["batch_size"])
        self.warmup_steps = int(p["warmup_steps"])
        self.updates_per_step = int(p["updates_per_step"])

        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, int(p["buffer_size"]))
        self.normalizer = (
            RunningNormalizer(self.obs_dim) if p.get("normalize_obs", True) else None
        )

        # Action bounds; replaced by the trainer with the environment's real
        # limits when they are available.
        self.act_low = -np.ones(self.act_dim, dtype=np.float32)
        self.act_high = np.ones(self.act_dim, dtype=np.float32)

        self.total_steps = 0
        self.updates = 0
        self._progress = 0.0
        self._critic_losses: list = []
        self._actor_losses: list = []

    # -- observation / action marshalling -------------------------------- #
    @staticmethod
    def _raw(obs) -> np.ndarray:
        return np.asarray(obs["obs"] if isinstance(obs, dict) else obs, dtype=np.float32).reshape(-1)

    def _obs(self, obs) -> np.ndarray:
        """The observation as the networks see it: normalised by the *current* statistics."""
        arr = self._raw(obs)
        return self.normalizer.normalize(arr) if self.normalizer is not None else arr

    def _to_env_action(self, a_norm: np.ndarray) -> np.ndarray:
        """Map a normalised ``[-1, 1]`` action onto the environment's range."""
        low, high = self.act_low, self.act_high
        env_a = low + (np.clip(a_norm, -1.0, 1.0) + 1.0) * 0.5 * (high - low)
        return env_a.astype(np.float32)

    def _to_norm_action(self, a_env) -> np.ndarray:
        """Inverse of :meth:`_to_env_action`, used when storing transitions."""
        low, high = self.act_low, self.act_high
        span = high - low
        span = np.where(np.abs(span) < 1e-12, 1.0, span)
        return (2.0 * (np.asarray(a_env, dtype=np.float32) - low) / span - 1.0).astype(
            np.float32
        )

    def _tensor(self, arr) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.dev)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(tau * sp.data)

    # -- agent interface -------------------------------------------------- #
    def set_training_progress(self, progress: float):
        self._progress = float(np.clip(progress, 0.0, 1.0))

    def update(self, obs, action, reward, next_obs, done):
        # Store RAW observations and normalise when sampling. Before v3.7 the
        # buffer held observations normalised with the statistics of the moment
        # they were stored; the first ones (statistics from a handful of steps)
        # stayed in the buffer inconsistent with everything after, and SAC, TD3,
        # DDPG and CrossQ failed to learn Pendulum within 40 episodes.
        s = self._raw(obs)
        s2 = self._raw(next_obs)
        if self.normalizer is not None and self.TRAIN:
            if self.normalizer.count == 0:
                self.normalizer.update(s)
            self.normalizer.update(s2)
        a = np.clip(self._to_norm_action(action), -1.0, 1.0)
        # Do not bootstrap past a true terminal state; a truncated episode
        # (time-limit) should still bootstrap.
        # The trainer sets last_step_truncated; before it existed, every time
        # limit (e.g. Pendulum's 200 steps) was stored as a true terminal.
        terminal = bool(done) and not getattr(self, "last_step_truncated", False)
        self.buffer.add(s, a, float(reward), s2, terminal)
        self.total_steps += 1

        if self.total_steps < self.warmup_steps:
            return
        if len(self.buffer) < self.batch_size:
            return
        for _ in range(self.updates_per_step):
            self._learn()

    def _learn(self):
        raise NotImplementedError

    def end_of_episode(self):
        pass

    def get_metrics(self) -> Dict[str, float]:
        metrics = {
            "buffer_size": len(self.buffer),
            "updates": self.updates,
            "steps": self.total_steps,
        }
        if self._critic_losses:
            metrics["critic_loss"] = float(np.mean(self._critic_losses[-500:]))
        if self._actor_losses:
            metrics["actor_loss"] = float(np.mean(self._actor_losses[-500:]))
        return metrics

    def _sample_batch(self):
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        if self.normalizer is not None:
            obs, next_obs = self.normalizer.normalize(obs), self.normalizer.normalize(next_obs)
        return (
            self._tensor(obs),
            self._tensor(act),
            self._tensor(rew),
            self._tensor(next_obs),
            self._tensor(done),
        )

    def _base_checkpoint(self) -> Dict[str, Any]:
        return {
            "agent_name": self.SPEC["name"],
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "params": self.params,
            "total_steps": self.total_steps,
            "updates": self.updates,
            "act_low": self.act_low.tolist(),
            "act_high": self.act_high.tolist(),
            "normalizer": self.normalizer.state_dict() if self.normalizer else None,
        }

    def _restore_base(self, ckpt: Dict[str, Any]) -> None:
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates = ckpt.get("updates", 0)
        if "act_low" in ckpt and "act_high" in ckpt:
            self.act_low = np.asarray(ckpt["act_low"], dtype=np.float32)
            self.act_high = np.asarray(ckpt["act_high"], dtype=np.float32)
        if self.normalizer is not None and ckpt.get("normalizer"):
            self.normalizer.load_state_dict(ckpt["normalizer"])


# --------------------------------------------------------------------------- #
# DDPG
# --------------------------------------------------------------------------- #

DDPG_PARAMS = _COMMON_PARAMS + [
    ParamSpec("expl_noise", "Exploration noise", 0.1, "float",
              "Std-dev of Gaussian noise added to actions during training. "
              "Measured in normalised action units.",
              min=0.0, max=2.0, recommended=0.1),
    ParamSpec("noise_decay", "Decay exploration noise", False, "bool",
              "Linearly anneal the exploration noise to zero over training.",
              advanced=True),
]

DDPG_PRESETS = [
    {"name": "quick-demo", "label": "Quick demo",
     "description": "Short run to confirm the pipeline works.",
     "overrides": {"warmup_steps": 500}, "episodes": 100},
    {"name": "beginner-default", "label": "Beginner default",
     "description": "Balanced defaults for Pendulum and InvertedPendulum.",
     "overrides": {}, "episodes": 300},
    {"name": "stable-baseline", "label": "Stable baseline",
     "description": "Lower learning rates and slower targets for reference curves.",
     "overrides": {"lr": 1e-4, "critic_lr": 1e-3, "tau": 0.002}, "episodes": 800},
    {"name": "fast-iteration", "label": "Fast iteration",
     "description": "Fewer episodes for quick hyperparameter sweeps.",
     "overrides": {"warmup_steps": 500}, "episodes": 150},
]


class DDPGAgent(_ContinuousBase):
    """Deep Deterministic Policy Gradient.

    A deterministic actor is trained by ascending the gradient of the critic:
    ``max_theta Q(s, mu_theta(s))``. Exploration comes from adding noise to the
    executed action rather than from a stochastic policy.
    """

    SPEC = build_spec(
        name="ddpg",
        label="DDPG",
        description="Off-policy actor-critic for continuous actions: a deterministic "
        "policy trained by ascending the critic's gradient, with a replay buffer "
        "and softly-updated target networks.",
        category="actor-critic",
        policy_type="off-policy",
        status="implemented",
        params=DDPG_PARAMS,
        presets=DDPG_PRESETS,
        action_types=["continuous"],
        obs_types=["vector"],
        beginner=(
            "DDPG keeps two networks. The 'critic' estimates how good a particular "
            "action is in a particular situation; the 'actor' directly outputs the "
            "action to take. The actor is trained by asking the critic 'which way "
            "should I change my output to get a higher score?' and following that "
            "gradient. Because the policy is deterministic, exploration has to be "
            "bolted on: DDPG adds random noise to the action it executes."
        ),
        technical=(
            "Critic minimises MSE(Q(s,a), r + gamma * Q_target(s2, mu_target(s2)) * "
            "(1-done)). Actor maximises Q(s, mu(s)). Both target networks are updated "
            "by Polyak averaging with coefficient tau every gradient step. Actions "
            "are stored in the buffer normalised to [-1,1] and rescaled to the "
            "environment's bounds when executed. Observations are optionally "
            "whitened with running mean/std."
        ),
        intuition=(
            "Instead of scoring every possible action and picking the best, learn a "
            "direct 'state -> action' mapping, and use the critic's gradient as the "
            "training signal for that mapping."
        ),
        update_rule=(
            r"y = r + γ · Q̄(s′, μ̄(s′)) · (1 − done);   "
            r"L_critic = (Q(s,a) − y)²;   L_actor = −Q(s, μ(s))"
        ),
        pseudocode=(
            "for each environment step:\n"
            "    a = mu(s) + noise                  # clipped to the action range\n"
            "    s2, r, done = env.step(a)\n"
            "    buffer.add(s, a, r, s2, done)\n"
            "    batch = buffer.sample(B)\n"
            "    y = r + gamma * Q_target(s2, mu_target(s2)) * (1-done)\n"
            "    update critic to minimise (Q(s,a) - y)^2\n"
            "    update actor to maximise Q(s, mu(s))\n"
            "    soft-update both target networks: target <- (1-tau)*target + tau*online"
        ),
        lifecycle=(
            "1. Act: deterministic policy output plus Gaussian exploration noise.\n"
            "2. Store the normalised transition in the replay buffer.\n"
            "3. After `warmup_steps`, take `updates_per_step` gradient steps per "
            "environment step.\n"
            "4. Soft-update the target actor and critic every gradient step.\n"
            "5. Optionally anneal the exploration noise across training."
        ),
        learns="A deterministic actor mu(s) and a critic Q(s,a).",
        exploration=(
            "Additive Gaussian noise with std `expl_noise`, applied in normalised "
            "action units; optionally annealed to zero over training."
        ),
        replay_buffer="Uniform sampling from the last `buffer_size` transitions.",
        target_network=(
            "Polyak soft update of both actor and critic every gradient step, rate "
            "`tau`."
        ),
        advantage="Not used - the critic provides the actor's gradient directly.",
        networks=(
            "Actor: obs -> 256 -> ReLU -> 256 -> ReLU -> tanh -> action. "
            "Critic: (obs, action) -> 256 -> ReLU -> 256 -> ReLU -> 1."
        ),
        failure_modes=[
            "Overestimates Q-values (no twin critics - this is what TD3 fixes)",
            "Very sensitive to learning rate and exploration noise scale",
            "Deterministic exploration can stall: if the noise is too small the "
            "agent never discovers better behaviour; too large and it never "
            "consolidates",
            "Can collapse to outputting constant extreme actions early in training",
        ],
        expected_metrics=[
            {"key": "critic_loss", "meaning": "MSE of TD targets - should settle low."},
            {"key": "actor_loss", "meaning": "-Q(s, mu(s)); becoming more negative means the actor is finding higher-value actions."},
            {"key": "buffer_size", "meaning": "Transitions stored so far."},
        ],
        related=["td3", "sac", "ddpg"],
        recommended_envs=[
            "inverted_pendulum", "reacher", "pendulum", "mountain_car_continuous"
        ],
        poor_envs=["humanoid", "swimmer", "ant"],
    )

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any],
                 device: str = "cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        p = self.params
        hidden = int(p["hidden_size"])
        self.actor = DeterministicActor(obs_dim, act_dim, hidden).to(self.dev)
        self.actor_target = DeterministicActor(obs_dim, act_dim, hidden).to(self.dev)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QCritic(obs_dim, act_dim, hidden).to(self.dev)
        self.critic_target = QCritic(obs_dim, act_dim, hidden).to(self.dev)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=p["lr"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=p["critic_lr"])

    def _current_noise(self) -> float:
        sigma = float(self.params["expl_noise"])
        if self.params.get("noise_decay", False):
            sigma *= max(0.0, 1.0 - self._progress)
        return sigma

    def get_action(self, obs, valid_actions=None):
        state = self._tensor(self._obs(obs)).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        if self.TRAIN:
            sigma = self._current_noise()
            if sigma > 0.0:
                action = action + np.random.normal(0.0, sigma, size=action.shape)
        return self._to_env_action(action)

    def _learn(self):
        obs, act, rew, next_obs, done = self._sample_batch()

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            y = rew + self.gamma * (1.0 - done) * target_q

        q = self.critic(obs, act)
        critic_loss = F.mse_loss(q, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.actor_target, self.actor, self.tau)

        self.updates += 1
        self._critic_losses.append(critic_loss.item())
        self._actor_losses.append(actor_loss.item())

    def get_metrics(self):
        m = super().get_metrics()
        m["expl_noise"] = round(self._current_noise(), 4)
        return m

    def save_model(self, path):
        torch.save(
            {
                **self._base_checkpoint(),
                "format": "ddpg",
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
        for name, optim in (("actor_optim", self.actor_optim),
                            ("critic_optim", self.critic_optim)):
            try:
                optim.load_state_dict(ckpt[name])
            except Exception:
                pass
        self._restore_base(ckpt)


# --------------------------------------------------------------------------- #
# TD3
# --------------------------------------------------------------------------- #

TD3_PARAMS = _COMMON_PARAMS + [
    ParamSpec("expl_noise", "Exploration noise", 0.1, "float",
              "Std-dev of Gaussian noise added to actions during training.",
              min=0.0, max=2.0, recommended=0.1),
    ParamSpec("policy_delay", "Policy delay", 2, "int",
              "Gradient steps between actor (and target) updates. Delaying them "
              "stops the policy chasing an unstable critic.",
              min=1, max=64),
    ParamSpec("target_noise", "Target policy noise", 0.2, "float",
              "Noise added to the target action when computing TD targets "
              "(target policy smoothing).", min=0.0, max=2.0, advanced=True),
    ParamSpec("noise_clip", "Target noise clip", 0.5, "float",
              "Target smoothing noise is clipped to +/- this value.",
              min=0.0, max=2.0, advanced=True),
    ParamSpec("network", "Network", "mlp", "choice",
              "mlp: ReLU layers. simba: residual pre-norm blocks with LayerNorm "
              "(Lee et al., 2025), which keep improving as they get wider. At the same "
              "width SimBa has about 8x the parameters, so it is slower on a CPU.",
              choices=["mlp", "simba"], advanced=True),
]

TD3_PRESETS = [
    {"name": "quick-demo", "label": "Quick demo",
     "description": "Short run to confirm the pipeline works.",
     "overrides": {"warmup_steps": 500}, "episodes": 100},
    {"name": "beginner-default", "label": "Beginner default",
     "description": "The reference TD3 configuration from the paper.",
     "overrides": {}, "episodes": 300},
    {"name": "stable-baseline", "label": "Stable baseline",
     "description": "Slower targets and more delayed updates for smoother curves.",
     "overrides": {"tau": 0.002, "policy_delay": 3}, "episodes": 800},
    {"name": "fast-iteration", "label": "Fast iteration",
     "description": "Fewer episodes for quick hyperparameter sweeps.",
     "overrides": {"warmup_steps": 500}, "episodes": 150},
]


class TD3Agent(_ContinuousBase):
    """Twin Delayed DDPG - DDPG plus three targeted stability fixes."""

    SPEC = build_spec(
        name="td3",
        label="TD3",
        description="Twin Delayed DDPG: clipped double Q-learning, delayed policy "
        "updates and target policy smoothing - three fixes that remove DDPG's "
        "overestimation bias and most of its hyperparameter fragility.",
        category="actor-critic",
        policy_type="off-policy",
        status="implemented",
        params=TD3_PARAMS,
        presets=TD3_PRESETS,
        action_types=["continuous"],
        obs_types=["vector"],
        beginner=(
            "TD3 is DDPG with three fixes for problems researchers kept hitting. "
            "(1) It trains two critics and always believes the more pessimistic one, "
            "which stops the agent fooling itself about how good an action is. "
            "(2) It updates the actor less often than the critic, so the policy does "
            "not chase a value estimate that is still moving. (3) When scoring the "
            "next state it adds a little noise to the action, which stops the agent "
            "exploiting narrow spikes in its own value estimate."
        ),
        technical=(
            "y = r + gamma * (1-done) * min_{i=1,2} Q_target_i(s2, a2), where "
            "a2 = clip(mu_target(s2) + clip(N(0, sigma), -c, c), -1, 1). The critic "
            "is updated every step; the actor and all target networks are updated "
            "only every `policy_delay` steps (the paper's delayed policy updates)."
        ),
        intuition=(
            "Be pessimistic about value, be patient about changing the policy, and "
            "smooth the target so a narrow spike in the critic cannot be exploited."
        ),
        update_rule=(
            r"a2 = clip(μ̄(s′) + clip(ε, −c, c), −1, 1),  ε ~ N(0, σ);"
            "\n"
            r"y = r + γ · (1 − done) · min_{i=1,2} Q̄_i(s′, a2)"
        ),
        pseudocode=(
            "for each environment step:\n"
            "    a = mu(s) + N(0, expl_noise)\n"
            "    s2, r, done = env.step(a)\n"
            "    buffer.add(s, a, r, s2, done)\n"
            "    batch = buffer.sample(B)\n"
            "    a2 = clip(mu_target(s2) + clip(N(0, target_noise), -c, c), -1, 1)\n"
            "    y = r + gamma * (1-done) * min(Q1_target(s2,a2), Q2_target(s2,a2))\n"
            "    update both critics towards y\n"
            "    if step % policy_delay == 0:\n"
            "        update actor to maximise Q1(s, mu(s))\n"
            "        soft-update all three target networks"
        ),
        lifecycle=(
            "1. Act with the deterministic policy plus Gaussian noise.\n"
            "2. Store the normalised transition.\n"
            "3. Update both critics every gradient step against the clipped "
            "double-Q target.\n"
            "4. Every `policy_delay` steps, update the actor and soft-sync the "
            "targets.\n"
            "5. Repeat; noise is not annealed by default."
        ),
        learns="A deterministic actor plus two critics (the min is used for targets).",
        exploration="Additive Gaussian noise with std `expl_noise` (not annealed).",
        replay_buffer="Uniform sampling from the last `buffer_size` transitions.",
        target_network=(
            "Polyak soft update of the actor and both critics, applied only on "
            "delayed (actor) update steps."
        ),
        advantage="Not used - twin critics replace the need for a baseline.",
        networks=(
            "Actor: obs -> 256 -> ReLU -> 256 -> ReLU -> tanh -> action. "
            "Two critics: (obs, action) -> 256 -> ReLU -> 256 -> ReLU -> 1."
        ),
        failure_modes=[
            "Still deterministic at evaluation time - on genuinely stochastic tasks "
            "a stochastic policy (SAC) can be better",
            "Too large `expl_noise` prevents the policy from ever consolidating",
            "Delayed updates mean slower adaptation if `policy_delay` is set high",
        ],
        expected_metrics=[
            {"key": "critic_loss", "meaning": "MSE of the clipped double-Q targets."},
            {"key": "actor_loss", "meaning": "-Q1(s, mu(s)); only updated on delayed steps."},
            {"key": "buffer_size", "meaning": "Transitions stored so far."},
        ],
        related=["ddpg", "sac"],
        recommended_envs=[
            "half_cheetah", "hopper", "walker2d", "ant", "inverted_double_pendulum"
        ],
        poor_envs=["swimmer"],
    )

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any],
                 device: str = "cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        p = self.params
        hidden = int(p["hidden_size"])
        net = p.get("network", "mlp")
        self.actor = DeterministicActor(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.actor_target = DeterministicActor(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TwinQCritic(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.critic_target = TwinQCritic(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=p["lr"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=p["critic_lr"])
        self.policy_delay = max(1, int(p["policy_delay"]))

    def get_action(self, obs, valid_actions=None):
        state = self._tensor(self._obs(obs)).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        if self.TRAIN:
            sigma = float(self.params["expl_noise"])
            if sigma > 0.0:
                action = action + np.random.normal(0.0, sigma, size=action.shape)
        return self._to_env_action(action)

    def _learn(self):
        obs, act, rew, next_obs, done = self._sample_batch()

        with torch.no_grad():
            noise = torch.randn_like(act) * self.params["target_noise"]
            noise = noise.clamp(-self.params["noise_clip"], self.params["noise_clip"])
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            y = rew + self.gamma * (1.0 - done) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        self.updates += 1
        self._critic_losses.append(critic_loss.item())

        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1(obs, self.actor(obs)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optim.step()
            self._actor_losses.append(actor_loss.item())

            self._soft_update(self.critic_target, self.critic, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)

    def save_model(self, path):
        torch.save(
            {
                **self._base_checkpoint(),
                "format": "td3",
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
        for name, optim in (("actor_optim", self.actor_optim),
                            ("critic_optim", self.critic_optim)):
            try:
                optim.load_state_dict(ckpt[name])
            except Exception:
                pass
        self._restore_base(ckpt)


# --------------------------------------------------------------------------- #
# SAC
# --------------------------------------------------------------------------- #

SAC_PARAMS = _COMMON_PARAMS + [
    ParamSpec("alpha", "Entropy temperature", 0.2, "float",
              "Weight on the entropy bonus when auto-tuning is off. Higher = more "
              "random, more exploratory policy.",
              min=0.0, max=10.0, recommended=0.2),
    ParamSpec("auto_entropy", "Auto-tune entropy", True, "bool",
              "Learn the temperature automatically so that the policy's entropy "
              "matches a target. This removes the single most annoying SAC "
              "hyperparameter.",
              advanced=True),
    ParamSpec("target_entropy_scale", "Target entropy scale", 1.0, "float",
              "Target entropy = -scale * action_dim. 1.0 is the standard "
              "recommendation; lower values encourage less randomness.",
              min=0.0, max=2.0, advanced=True),
    ParamSpec("alpha_lr", "Temperature learning rate", 3e-4, "float",
              "Adam learning rate for the entropy temperature.",
              min=1e-6, max=1.0, advanced=True),
    ParamSpec("network", "Network", "mlp", "choice",
              "mlp: ReLU layers. simba: residual pre-norm blocks with LayerNorm "
              "(Lee et al., 2025), which keep improving as they get wider. At the same "
              "width SimBa has about 8x the parameters, so it is slower on a CPU.",
              choices=["mlp", "simba"], advanced=True),
]

SAC_PRESETS = [
    {"name": "quick-demo", "label": "Quick demo",
     "description": "Short run to confirm the pipeline works.",
     "overrides": {"warmup_steps": 500}, "episodes": 100},
    {"name": "beginner-default", "label": "Beginner default",
     "description": "Auto-tuned entropy - the best starting point for a new task.",
     "overrides": {}, "episodes": 300},
    {"name": "stable-baseline", "label": "Stable baseline",
     "description": "Slower targets and a larger batch for reference curves.",
     "overrides": {"tau": 0.002, "batch_size": 512}, "episodes": 800},
    {"name": "fast-iteration", "label": "Fast iteration",
     "description": "Fewer episodes for quick hyperparameter sweeps.",
     "overrides": {"warmup_steps": 500}, "episodes": 150},
]


class SACAgent(_ContinuousBase):
    """Soft Actor-Critic - maximum-entropy off-policy actor-critic."""

    SPEC = build_spec(
        name="sac",
        label="SAC",
        description="Maximum-entropy actor-critic: the policy is trained to be as "
        "random as possible while still collecting high reward. With automatic "
        "temperature tuning this is usually the most robust choice for a new "
        "continuous-control task.",
        category="actor-critic",
        policy_type="off-policy",
        status="implemented",
        params=SAC_PARAMS,
        presets=SAC_PRESETS,
        action_types=["continuous"],
        obs_types=["vector"],
        beginner=(
            "SAC changes the objective: instead of only maximising reward, it "
            "maximises reward *plus randomness*. Staying as unpredictable as "
            "possible while still doing well forces the agent to keep trying "
            "alternatives instead of committing too early, which turns out to be a "
            "very effective form of exploration. Best of all, the amount of "
            "randomness to aim for is learned automatically, so there is one fewer "
            "knob to tune."
        ),
        technical=(
            "Policy is a squashed Gaussian sampled with the reparameterisation "
            "trick. Critic targets use the clipped double-Q minimum minus the "
            "entropy bonus: y = r + gamma*(1-done)*(min_i Q_target_i(s2,a2) - "
            "alpha*log pi(a2|s2)) with a2 ~ pi(.|s2). The actor minimises "
            "alpha*log pi(a|s) - min_i Q_i(s,a). With `auto_entropy` the "
            "temperature alpha is learned by minimising "
            "-alpha*(log pi + H_target), H_target = -target_entropy_scale * act_dim."
        ),
        intuition=(
            "Reward the policy for being uncertain. A policy that is forced to stay "
            "somewhat random keeps exploring, and 'good but random' is a much "
            "easier target to hit than a single perfect deterministic answer."
        ),
        update_rule=(
            r"y = r + γ(1−done)·( min_i Q̄_i(s′,a′) − α·log π(a′|s′) ),  a′ ~ π(·|s′)"
            "\n"
            r"L_actor = α·log π(a|s) − min_i Q_i(s,a);   "
            r"L_α = −α·(log π(a|s) + H_target)"
        ),
        pseudocode=(
            "for each environment step:\n"
            "    a ~ pi(.|s)                        # tanh-Gaussian sample\n"
            "    s2, r, done = env.step(a)\n"
            "    buffer.add(s, a, r, s2, done)\n"
            "    batch = buffer.sample(B)\n"
            "    a2, logp2 = pi(.|s2)               # with reparameterisation\n"
            "    y = r + gamma*(1-done)*(min(Q1t,Q2t)(s2,a2) - alpha*logp2)\n"
            "    update both critics towards y\n"
            "    a_new, logp = pi(.|s)\n"
            "    update actor to maximise min(Q1,Q2)(s,a_new) - alpha*logp\n"
            "    if auto_entropy: update alpha towards target entropy\n"
            "    soft-update both target critics (SAC has no target actor)"
        ),
        lifecycle=(
            "1. Sample an action from the stochastic policy (or use tanh(mu) when "
            "evaluating).\n"
            "2. Store the normalised transition.\n"
            "3. After `warmup_steps`, take `updates_per_step` gradient steps per "
            "environment step: critics, then actor, then temperature.\n"
            "4. Soft-update the two target critics every step.\n"
            "5. There is no target actor - the bootstrap uses a freshly sampled "
            "action from the current policy."
        ),
        learns="A Gaussian actor (mean and log-std per action dim) plus two critics.",
        exploration=(
            "Intrinsic: the policy is stochastic and is explicitly rewarded for "
            "entropy. No hand-tuned noise schedule is needed."
        ),
        replay_buffer="Uniform sampling from the last `buffer_size` transitions.",
        target_network=(
            "Polyak soft update of the two target critics; unlike DDPG/TD3 there is "
            "no target actor."
        ),
        advantage="Not used - the entropy term acts as a learned, state-dependent bonus.",
        networks=(
            "Actor: shared trunk obs -> 256 -> ReLU -> 256 -> ReLU, then separate "
            "mu and log-std heads, squashed with tanh. "
            "Two critics: (obs, action) -> 256 -> ReLU -> 256 -> ReLU -> 1."
        ),
        failure_modes=[
            "With auto_entropy off, `alpha` becomes the critical hyperparameter and "
            "a bad value silently ruins learning",
            "The log-std head can saturate at its clamp bounds on some tasks",
            "Stochastic evaluation is noisier than a deterministic policy - the "
            "testbed evaluates with the mean action (tanh(mu)) for this reason",
        ],
        expected_metrics=[
            {"key": "critic_loss", "meaning": "MSE of entropy-adjusted TD targets."},
            {"key": "actor_loss", "meaning": "alpha*log_pi - min Q; should trend down."},
            {"key": "alpha", "meaning": "Current entropy temperature (learned when auto-tuning)."},
            {"key": "entropy", "meaning": "Mean policy log-probability; more negative = more deterministic."},
        ],
        related=["td3", "ddpg", "ppo"],
        recommended_envs=[
            "half_cheetah", "hopper", "walker2d", "ant", "humanoid",
            "inverted_pendulum", "reacher", "swimmer", "mountain_car_continuous"
        ],
    )

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any],
                 device: str = "cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        p = self.params
        hidden = int(p["hidden_size"])

        net = p.get("network", "mlp")
        self.actor = GaussianActor(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.critic = TwinQCritic(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.critic_target = TwinQCritic(obs_dim, act_dim, hidden, network=net).to(self.dev)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=p["lr"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=p["critic_lr"])

        self.auto_entropy = bool(p.get("auto_entropy", True))
        if self.auto_entropy:
            self.target_entropy = -float(p["target_entropy_scale"]) * self.act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.dev)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=p["alpha_lr"])
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optim = None
            self.alpha = float(p["alpha"])

        self._entropies: list = []
        self._alphas: list = []

    def get_action(self, obs, valid_actions=None):
        state = self._tensor(self._obs(obs)).unsqueeze(0)
        with torch.no_grad():
            if self.TRAIN:
                action, _ = self.actor.sample(state)
                action = action.squeeze(0).cpu().numpy()
            else:
                # Evaluate with the mean action: deterministic and higher-scoring.
                action = self.actor.deterministic(state).squeeze(0).cpu().numpy()
        return self._to_env_action(action)

    def _learn(self):
        obs, act, rew, next_obs, done = self._sample_batch()

        # --- critic --------------------------------------------------- #
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            # sample() returns log-probs shaped (B, 1) while the critics return
            # (B,). Without the squeeze the target broadcast to (B, B) and every
            # Q-value was regressed toward the batch-average target.
            next_logp = next_logp.squeeze(-1)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            min_q_t = torch.min(q1_t, q2_t)
            y = rew + self.gamma * (1.0 - done) * (min_q_t - self.alpha * next_logp)

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        # --- actor ---------------------------------------------------- #
        new_action, logp = self.actor.sample(obs)
        logp = logp.squeeze(-1)
        q1_new, q2_new = self.critic(obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * logp - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        # --- temperature ----------------------------------------------- #
        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().item())

        self._soft_update(self.critic_target, self.critic, self.tau)

        self.updates += 1
        self._critic_losses.append(critic_loss.item())
        self._actor_losses.append(actor_loss.item())
        self._entropies.append(float(-logp.mean().item()))
        self._alphas.append(self.alpha)

    def get_metrics(self):
        m = super().get_metrics()
        m["alpha"] = round(float(self.alpha), 5)
        if self._entropies:
            m["entropy"] = round(float(np.mean(self._entropies[-500:])), 4)
        return m

    def save_model(self, path):
        torch.save(
            {
                **self._base_checkpoint(),
                "format": "sac",
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
                "alpha_optim": self.alpha_optim.state_dict() if self.alpha_optim else None,
                "auto_entropy": self.auto_entropy,
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(
            ckpt.get("critic_target", ckpt["critic"])
        )
        for name, optim in (("actor_optim", self.actor_optim),
                            ("critic_optim", self.critic_optim)):
            try:
                optim.load_state_dict(ckpt[name])
            except Exception:
                pass
        if self.auto_entropy and ckpt.get("log_alpha") is not None:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.dev))
            self.alpha = float(self.log_alpha.exp().item())
            if self.alpha_optim and ckpt.get("alpha_optim"):
                try:
                    self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
                except Exception:
                    pass
        self._restore_base(ckpt)
