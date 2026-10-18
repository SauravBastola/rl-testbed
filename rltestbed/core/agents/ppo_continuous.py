"""PPO with a diagonal Gaussian policy, for continuous actions.

Reuses the discrete PPO's rollout buffer, GAE and clipped update loop
(:class:`.ppo.PPOAgent`) and replaces only the policy: a mean network plus a
state-independent learned log standard deviation. The policy acts in a
normalised [-1, 1] space; actions are rescaled to the environment's bounds.
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .base import ParamSpec, build_spec
from .continuous import RunningNormalizer
from .ppo import PARAMS as _PPO_PARAMS
from .ppo import PPOAgent


class GaussianPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, log_std_init: float):
        super().__init__()

        def mlp(out):
            return nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, out),
            )

        self.mean = mlp(act_dim)
        self.value = mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        # small final layer so the initial policy is centred with the chosen std
        nn.init.uniform_(self.mean[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean[-1].bias)

    def dist(self, x):
        return torch.distributions.Normal(self.mean(x), self.log_std.clamp(-5.0, 2.0).exp())

    def forward(self, x):
        return self.dist(x), self.value(x).squeeze(-1)


_PARAMS = [p for p in _PPO_PARAMS if p.name not in ("entropy_coef",)] + [
    ParamSpec("entropy_coef", "Entropy coefficient", 0.0, "float",
              "Exploration bonus; continuous PPO usually relies on the learned std instead.",
              min=0.0, max=1.0),
    ParamSpec("log_std_init", "Initial log std", -0.5, "float",
              "Starting exploration noise in normalised action units (std = e^value).",
              min=-5.0, max=2.0),
    ParamSpec("normalize_obs", "Normalise observations", True, "bool",
              "Running mean/std normalisation of observations; important for MuJoCo."),
]


class GaussianPPOAgent(PPOAgent):
    SPEC = build_spec(
        name="ppo_continuous",
        label="PPO (Gaussian policy)",
        description="PPO for continuous actions: a Gaussian policy with learned "
        "standard deviation, the same clipped update and GAE advantages.",
        category="actor-critic",
        policy_type="on-policy",
        action_types=["continuous"],
        obs_types=["vector", "onehot"],
        params=_PARAMS,
        presets=[
            {"name": "beginner-default", "label": "Beginner default",
             "description": "Standard settings for low-dimensional control.",
             "overrides": {}, "episodes": 300},
            {"name": "mujoco", "label": "MuJoCo", "description": "Larger rollouts and nets.",
             "overrides": {"rollout_steps": 2048, "hidden_size": 256, "minibatch_size": 64},
             "episodes": 500},
        ],
        beginner=(
            "The same PPO as for button presses, but each action is a number on a "
            "dial. The policy proposes a most likely setting and a spread around it; "
            "training moves the setting toward what worked and narrows the spread as "
            "the agent becomes confident."
        ),
        technical=(
            "π(a|s) = N(μ_θ(s), diag(σ²)) with log σ a free parameter. The clipped "
            "surrogate, GAE(λ) and value loss are identical to discrete PPO; the "
            "log-probability is summed over action dimensions. Actions are sampled "
            "in [-1, 1] and affinely rescaled to the environment's bounds; the "
            "unclipped sample is stored so the log-probability stays exact."
        ),
        intuition="Start with a wide, uncertain dial; let evidence narrow it.",
        update_rule="L = E[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)],  r_t = N(a_t; μ_θ(s_t), σ_θ) / N(a_t; μ_old, σ_old)",
        pseudocode=(
            "for each rollout of N steps:\n"
            "    a ~ Normal(mu(s), exp(log_std));  env.step(rescale(clip(a, -1, 1)))\n"
            "    compute GAE advantages and returns\n"
            "    for epoch in 1..K, for each minibatch:\n"
            "        ratio = exp(logp_new(a) - logp_old(a))\n"
            "        loss = -min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) + c*(V - R)^2"
        ),
        learns="A Gaussian policy (mean network and log std) and a state-value critic.",
        exploration="Sampling from the Gaussian; σ is learned and usually shrinks.",
        replay_buffer="Not used: on-policy.",
        target_network="Not used.",
        advantage="GAE(λ) with the learned critic.",
        networks="Two tanh MLPs (policy mean, value) plus a learned log-std vector.",
        failure_modes=[
            "Sample-hungry: needs far more environment steps than SAC or TD3",
            "σ collapsing early stops exploration; σ staying large keeps returns noisy",
            "Without observation normalisation MuJoCo returns often stall",
        ],
        expected_metrics=[
            {"key": "policy_loss", "meaning": "Clipped surrogate loss."},
            {"key": "value_loss", "meaning": "Critic MSE against GAE returns."},
            {"key": "action_std", "meaning": "Current exploration std in normalised units."},
            {"key": "approx_kl", "meaning": "Policy change per update; keep under about 0.02."},
        ],
        recommended_envs=["probe6", "pendulum", "mountain_car_continuous", "inverted_pendulum"],
        poor_envs=["cartpole", "gridworld"],
        related=["ppo", "sac", "td3"],
        paper="Schulman et al., 2017. Proximal Policy Optimization Algorithms.",
    )

    _ACT_DTYPE = np.float32

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        dev = torch.device(device)
        self.net = GaussianPolicyNet(obs_dim, act_dim, self.params["hidden_size"],
                                     self.params["log_std_init"]).to(dev)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        self.normalizer = RunningNormalizer(obs_dim) if self.params["normalize_obs"] else None
        self.act_low = -np.ones(act_dim, dtype=np.float32)
        self.act_high = np.ones(act_dim, dtype=np.float32)

    def set_action_bounds(self, low, high):
        super().set_action_bounds(low, high)
        self.act_low = np.asarray(low, dtype=np.float32).flatten()
        self.act_high = np.asarray(high, dtype=np.float32).flatten()

    def _prep(self, obs_raw):
        x = np.asarray(obs_raw, dtype=np.float32).flatten()
        if self.normalizer is not None:
            if self.TRAIN:
                self.normalizer.update(x)
            x = self.normalizer.normalize(x)
        return x

    def _rescale(self, a):
        a = np.clip(a, -1.0, 1.0)
        return self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)

    def get_action(self, obs, valid_actions=None):
        obs_raw, _ = self._unpack(obs)
        x = self._prep(obs_raw)
        state = torch.as_tensor(x, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.net(state)
            # greedy evaluation uses the mean, matching the trainer's greedy eval
            a = dist.sample() if self.TRAIN else dist.mean
            logp = dist.log_prob(a).sum(-1).item()
        a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
        if self.TRAIN:
            self._obs_buf.append(x)
            self._act_buf.append(a_np)
            self._logp_buf.append(logp)
            self._val_buf.append(float(value.item()))
        return self._rescale(a_np)

    def _evaluate(self, obs, acts, masks):
        dist, values = self.net(obs)
        return dist.log_prob(acts).sum(-1), dist.entropy().sum(-1).mean(), values

    def get_metrics(self):
        m = super().get_metrics()
        m.pop("entropy", None)
        m["action_std"] = round(float(self.net.log_std.exp().mean().item()), 4)
        return m

    def save_model(self, path):
        torch.save({
            "format": "ppo-gaussian-v1",
            "agent_name": self.SPEC["name"],
            "net": self.net.state_dict(),
            "normalizer": self.normalizer.state_dict() if self.normalizer else None,
            "params": self.params,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "act_low": self.act_low.tolist(),
            "act_high": self.act_high.tolist(),
        }, str(path))

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        if self.normalizer is not None and ckpt.get("normalizer"):
            self.normalizer.load_state_dict(ckpt["normalizer"])
        if ckpt.get("act_low") is not None:
            self.act_low = np.asarray(ckpt["act_low"], dtype=np.float32)
            self.act_high = np.asarray(ckpt["act_high"], dtype=np.float32)
