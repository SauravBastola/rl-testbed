"""REINFORCE (Monte-Carlo policy gradient) with optional learned baseline."""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Any, Dict, List

from .base import Agent, ParamSpec, build_spec

_COMMON = [
    ParamSpec("lr", "Learning rate", 1e-3, "float", "Adam learning rate for the policy",
              min=1e-6, max=1.0),
    ParamSpec("gamma", "Discount factor", 0.99, "float",
              "Discount applied when computing Monte-Carlo returns", min=0.0, max=0.9999),
    ParamSpec("hidden_size", "Hidden units", 128, "int", "Width of the two hidden layers",
              min=8, max=4096, advanced=True),
    ParamSpec("entropy_coef", "Entropy coefficient", 0.01, "float",
              "Bonus for policy entropy - keeps exploration alive", min=0.0, max=1.0,
              advanced=True),
]


def _spec(name: str, label: str, description: str, baseline: bool) -> Dict[str, Any]:
    params = list(_COMMON)
    if baseline:
        params.append(ParamSpec("baseline_lr", "Baseline lr", 1e-3, "float",
                                "Learning rate for the value baseline"))
    return build_spec(
        name=name,
        label=label,
        description=description,
        category="policy-gradient",
        policy_type="on-policy",
        status="implemented",
        # pixel observations are flattened for the MLP; check_compat warns about it
        obs_types=["vector", "onehot", "image"],
        params=params,
        presets=[
            {"name": "quick-demo", "label": "Quick demo",
             "description": "Short run that learns CartPole basics.",
             "overrides": {"lr": 2e-3}, "episodes": 200},
            {"name": "beginner-default", "label": "Beginner default",
             "description": "Balanced defaults for classic control tasks.",
             "overrides": {}, "episodes": 500},
            {"name": "stable-baseline", "label": "Stable baseline",
             "description": "Lower lr for smoother curves on longer runs.",
             "overrides": {"lr": 5e-4}, "episodes": 800},
            {"name": "fast-iteration", "label": "Fast iteration",
             "description": "Quick sweeps of lr and entropy.",
             "overrides": {}, "episodes": 150},
        ],
        beginner=(
            "REINFORCE is the simplest policy-gradient method: it directly nudges the "
            "probability of actions that led to good episode returns. After every full "
            "episode it looks at everything that happened and pushes probabilities up "
            "for good trajectories and down for bad ones."
            if not baseline else
            "This is REINFORCE plus a learned value baseline: instead of comparing a return "
            "to zero, each action's advantage is (return − predicted value). Subtracting the "
            "baseline does not change what is optimal but massively reduces gradient noise."
        ),
        technical=(
            "Monte-Carlo returns G_t are computed by backward discounting; returns are then "
            "standardised per episode. The loss is −Σ log π(a_t|s_t)·Ĝ_t plus an entropy "
            "bonus (−entropy_coef · H[π]). Adam, grad-norm clip 10."
            if not baseline else
            "Advantage Â_t = G_t − V_φ(s_t). Policy loss −Σ log π·Â̂ (advantages standardised); "
            "value loss 0.5·MSE(V_φ(s_t), G_t). Separate Adam optimisers for policy and "
            "baseline; grad-norm clip 10."
        ),
        intuition=(
            "Think of the policy as a habit strength for each action in each situation. "
            "REINFORCE reinforces habits that were followed by high reward - literally "
            "'reinforcement' learning."
        ),
        update_rule=(
            r"θ ← θ + α · Σ_t ∇θ log π_θ(a_t|s_t) · G_t" if not baseline
            else r"θ ← θ + α · Σ_t ∇θ log π_θ(a_t|s_t) · (G_t − V_φ(s_t))"
        ),
        pseudocode=(
            "for each episode:\n"
            "    trajectory = []\n"
            "    roll out π until done\n"
            "    G = 0\n"
            "    for (s,a,r) in reversed(trajectory):\n"
            "        G = r + gamma * G\n"
            "        store (s, a, G)\n"
            "    normalise returns\n"
            "    loss = -Σ log π(a|s) * G - c*H[π]\n"
            "    update θ with one gradient step"
            if not baseline else
            "for each episode:\n"
            "    roll out π, storing states/actions/rewards\n"
            "    compute discounted returns G_t\n"
            "    values V(s_t), advantages A = G - V\n"
            "    normalise advantages\n"
            "    policy_loss = -Σ log π(a|s)*A\n"
            "    value_loss  = MSE(V(s), G)\n"
            "    update θ and φ separately"
        ),
        lifecycle=(
            "1. Sample a complete episode with the current stochastic policy.\n"
            "2. At episode end compute discounted Monte-Carlo returns.\n"
            "3. Take one gradient step (policy + entropy bonus).\n"
            "4. Repeat. There is no replay buffer or target network."
        ),
        learns="A softmax policy over discrete actions (and a state-value baseline if enabled).",
        exploration="Inherent sampling from the softmax policy + entropy bonus.",
        replay_buffer="Not used - strictly on-policy, data is discarded after one update.",
        target_network="Not used.",
        advantage=(
            "None (raw returns)" if not baseline
            else "Learned baseline V(s); advantage = return − baseline"
        ),
        networks=(
            "Policy MLP: obs → hidden → ReLU → hidden → ReLU → act_dim logits."
            if not baseline else
            "Policy MLP as plain REINFORCE plus a value head MLP outputting a scalar."
        ),
        failure_modes=[
            "Very high variance without a baseline - curves are noisy",
            "Entropy collapse: policy becomes deterministic early and stops exploring",
            "Sensitive to lr; too high destroys the policy in a few updates",
            "Needs many episodes because each episode yields only ONE gradient step",
        ]
        if not baseline else
        [
            "Baseline can lag behind fast-improving policies early in training",
            "Still one gradient step per episode - sample inefficient vs PPO",
        ],
        expected_metrics=[
            {"key": "policy_loss", "meaning": "Policy-gradient loss magnitude."},
            {"key": "episode_return_ma", "meaning": "Running mean of raw episode returns."},
            *([] if not baseline else [{"key": "baseline_loss", "meaning": "Value-function MSE."}]),
        ],
        related=["reinforce_baseline"] if not baseline else ["reinforce", "a2c", "ppo"],
    )


class ReinforceAgent(Agent):
    """REINFORCE with Monte-Carlo discounted returns and entropy regularization."""

    SPEC = _spec(
        "reinforce", "REINFORCE",
        "Canonical policy-gradient method: learns from complete episodes using "
        "discounted Monte-Carlo returns with an entropy bonus.",
        baseline=False,
    )

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.obs_dim = obs_dim
        dev = torch.device(device)
        self.policy = self._make_policy(obs_dim, act_dim, params["hidden_size"]).to(dev)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=params["lr"])
        self.gamma = params["gamma"]
        self.entropy_coef = params["entropy_coef"]

        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self.episode_returns = deque(maxlen=100)
        self.policy_losses = deque(maxlen=500)

    @staticmethod
    def _make_policy(obs_dim, act_dim, hidden_size):
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def _dist(self, obs, mask=None):
        state = torch.as_tensor(np.asarray(obs, dtype=np.float32).flatten(),
                                device=self.device).unsqueeze(0)
        logits = self.policy(state).squeeze(0)
        if mask is not None:
            mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=self.device)
            logits = logits.masked_fill(~mask_t, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, obs, valid_actions=None):
        obs_raw, mask_list = self._unpack(obs)
        mask = mask_list
        if mask is None and valid_actions is not None:
            mask = [1 if i in valid_actions else 0 for i in range(self.act_dim)]
        dist = self._dist(obs_raw, mask)
        # Evaluation (TRAIN=False) is greedy, matching the trainer\'s "greedy eval".
        action = dist.sample() if self.TRAIN else dist.probs.argmax(dim=-1)
        if self.TRAIN:
            self._log_probs.append(dist.log_prob(action))
        return int(action.item())

    def update(self, obs, action, reward, next_obs, done):
        self._rewards.append(float(reward))

    def end_of_episode(self):
        if not self.TRAIN or not self._log_probs:
            self._reset_episode()
            return
        returns_t = self._compute_returns()
        if self._states_stored:
            # subclass hook (baseline variant) handles its own update
            self._update_with_baseline(returns_t)
        else:
            returns = returns_t
            if returns.numel() >= 8 and returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            log_probs = torch.stack(self._log_probs)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum()
            loss = -(log_probs * returns).sum() - self.entropy_coef * entropy

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.optim.step()

            self.policy_losses.append(loss.item())
            self.episode_returns.append(sum(self._rewards))
        self._reset_episode()

    #: subclasses may stash states
    _states_stored = False

    def _compute_returns(self) -> torch.Tensor:
        returns, G = [], 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def _update_with_baseline(self, returns_t):  # pragma: no cover - overridden
        raise NotImplementedError

    def _reset_episode(self):
        self._log_probs = []
        self._rewards = []

    def get_metrics(self):
        return {
            "policy_loss": float(np.mean(self.policy_losses)) if self.policy_losses else 0.0,
            "episode_return_ma": float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
        }

    def save_model(self, path):
        torch.save(
            {
                "format": type(self).__name__,
                "agent_name": self.SPEC["name"],
                "policy": self.policy.state_dict(),
                "params": self.params,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])


class ReinforceBaselineAgent(ReinforceAgent):
    """REINFORCE with a learned value baseline (advantage = return - V(s))."""

    _states_stored = True

    SPEC = _spec(
        "reinforce_baseline", "REINFORCE + Baseline",
        "REINFORCE with a learned state-value baseline; the advantage "
        "(return - baseline) reduces gradient variance and speeds learning.",
        baseline=True,
    )

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        dev = torch.device(device)
        self.baseline = nn.Sequential(
            nn.Linear(obs_dim, params["hidden_size"]), nn.ReLU(),
            nn.Linear(params["hidden_size"], 1),
        ).to(dev)
        self.baseline_optim = torch.optim.Adam(
            self.baseline.parameters(), lr=params["baseline_lr"]
        )
        self._states: List[torch.Tensor] = []
        self.baseline_losses = deque(maxlen=500)

    def get_action(self, obs, valid_actions=None):
        obs_raw, _ = self._unpack(obs)
        if self.TRAIN:
            self._states.append(
                torch.as_tensor(np.asarray(obs_raw, dtype=np.float32).flatten(),
                                device=self.device)
            )
        return super().get_action(obs_raw, valid_actions)

    def _update_with_baseline(self, returns_t):
        states = torch.stack(self._states[: returns_t.numel()])
        values = self.baseline(states).squeeze(-1)
        advantages = returns_t - values.detach()
        if advantages.numel() >= 8 and advantages.std() > 1e-8:  # see A2C
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(self._log_probs)
        policy_loss = -(log_probs * advantages).sum()
        value_loss = nn.functional.mse_loss(values, returns_t)
        loss = policy_loss + 0.5 * value_loss

        self.optim.zero_grad()
        self.baseline_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optim.step()
        self.baseline_optim.step()

        self.policy_losses.append(policy_loss.item())
        self.baseline_losses.append(value_loss.item())
        self.episode_returns.append(sum(self._rewards))

    def _reset_episode(self):
        super()._reset_episode()
        self._states = []

    def get_metrics(self):
        m = super().get_metrics()
        m["baseline_loss"] = (
            float(np.mean(self.baseline_losses)) if self.baseline_losses else 0.0
        )
        return m

    def save_model(self, path):
        torch.save(
            {
                "format": type(self).__name__,
                "agent_name": self.SPEC["name"],
                "policy": self.policy.state_dict(),
                "baseline": self.baseline.state_dict(),
                "params": self.params,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        if "baseline" in ckpt:
            self.baseline.load_state_dict(ckpt["baseline"])
