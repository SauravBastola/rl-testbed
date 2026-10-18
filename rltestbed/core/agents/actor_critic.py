"""Actor-Critic (A2C) for discrete action spaces.

One gradient update per episode-segment (here: per episode end) combining a
policy-gradient objective with a learned value baseline and entropy bonus -
the bridge from REINFORCE to PPO.
"""

from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .base import Agent, ParamSpec, build_spec


class ActorCriticNet(nn.Module):
    """Shared-trunk MLP with separate policy and value heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


PARAMS = [
    ParamSpec("lr", "Learning rate", 3e-4, "float", "Adam learning rate (shared network)",
              min=1e-6, max=1.0),
    ParamSpec("gamma", "Discount factor", 0.99, "float",
              "Discount for returns and bootstrapped values", min=0.0, max=0.9999),
    ParamSpec("hidden_size", "Hidden units", 128, "int", "Width of shared hidden layers",
              min=8, max=4096, advanced=True),
    ParamSpec("entropy_coef", "Entropy coefficient", 0.01, "float",
              "Exploration bonus added to the policy loss", min=0.0, max=1.0),
    ParamSpec("value_coef", "Value loss coef", 0.5, "float",
              "Weight of the value-function loss relative to the policy loss",
              min=0.0, max=10.0, advanced=True),
    ParamSpec("max_grad_norm", "Grad clip", 0.5, "float", "Max gradient norm",
              min=0.05, max=100.0, advanced=True),
]

_SPEC = build_spec(
    name="a2c",
    label="Actor-Critic (A2C)",
    description="Advantage actor-critic with a shared trunk: learns both a policy "
    "(actor) and a value function (critic); lower variance than REINFORCE.",
    category="actor-critic",
    policy_type="on-policy",
    status="implemented",
    # pixel observations are flattened for the MLP; check_compat warns about it
    obs_types=["vector", "onehot", "image"],
    params=PARAMS,
    presets=[
        {"name": "beginner-default", "label": "Beginner default",
         "description": "Reliable settings for CartPole-class tasks.",
         "overrides": {}, "episodes": 400},
        {"name": "quick-demo", "label": "Quick demo",
         "description": "Fast learning signal on easy environments.",
         "overrides": {"lr": 7e-4}, "episodes": 150},
        {"name": "stable-baseline", "label": "Stable baseline",
         "description": "Conservative lr for smooth reference curves.",
         "overrides": {"lr": 1e-4}, "episodes": 800},
        {"name": "fast-iteration", "label": "Fast iteration",
         "description": "Short runs to probe entropy/value coefficients.",
         "overrides": {}, "episodes": 200},
    ],
    beginner=(
        "A2C runs two learners on one brain: an 'actor' that picks actions and a 'critic' "
        "that estimates how good each situation is. The actor improves using the critic's "
        "verdict (advantage = outcome − expectation), so credit assignment is much less "
        "noisy than plain REINFORCE."
    ),
    technical=(
        "Monte-Carlo discounted returns G_t; advantages Â_t = G_t − V(s_t) standardised per "
        "episode. Loss = policy_loss + value_coef·value_loss − entropy_coef·H[π], where "
        "policy_loss = −Σ log π(a_t|s_t)·Â_t and value_loss = MSE(V(s_t), G_t). Shared-trunk "
        "MLP, Adam, grad-norm clip at max_grad_norm."
    ),
    intuition=(
        "The critic turns 'was the total reward good?' into 'was this move better than I "
        "expected?'. Learning from surprise rather than raw score is what makes actor-critic "
        "methods sample-efficient."
    ),
    update_rule=r"θ ← θ + α Σ ∇θ log π(a_t|s_t)·Â_t ,   Â_t ≈ G_t − V_φ(s_t)",
    pseudocode=(
        "for each episode:\n"
        "    roll out π, store states/actions/rewards\n"
        "    G_t = discounted returns\n"
        "    logits, V = net(states)\n"
        "    A = G - V.detach();  A = normalise(A)\n"
        "    loss = -(logπ(a|s)*A).sum()\n"
        "           + value_coef * mse(V, G)\n"
        "           - entropy_coef * H[π]\n"
        "    one Adam step on all parameters"
    ),
    lifecycle=(
        "1. Collect a full episode with the stochastic policy.\n"
        "2. Compute Monte-Carlo returns.\n"
        "3. One joint gradient step on actor + critic losses (+ entropy bonus).\n"
        "4. Repeat."
    ),
    learns="A softmax policy over actions and a state-value function sharing one MLP trunk.",
    exploration="Stochastic policy sampling + entropy regularisation.",
    replay_buffer="Not used - strictly on-policy.",
    target_network="Not used.",
    advantage="Learned critic: advantage = return − V(s), standardised per episode.",
    networks="Shared MLP trunk → linear policy head (act_dim logits) + linear value head (scalar).",
    failure_modes=[
        "Policy can collapse if entropy_coef is too small",
        "Shared trunk means value-learning gradients can disturb the policy (tune value_coef)",
        "Still one update per episode - slow on very long episodes",
    ],
    expected_metrics=[
        {"key": "policy_loss", "meaning": "Actor loss magnitude."},
        {"key": "value_loss", "meaning": "Critic MSE - should decrease as the critic calibrates."},
        {"key": "entropy", "meaning": "Policy entropy - watch for premature collapse."},
    ],
    related=["reinforce_baseline", "ppo"],
)


class A2CAgent(Agent):
    SPEC = _SPEC

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.obs_dim = obs_dim
        dev = torch.device(device)
        self.net = ActorCriticNet(obs_dim, act_dim, params["hidden_size"]).to(dev)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=params["lr"])
        self.gamma = params["gamma"]
        self.entropy_coef = params["entropy_coef"]
        self.value_coef = params["value_coef"]
        self.max_grad_norm = params["max_grad_norm"]

        self._log_probs = []
        self._actions = []
        self._masks = []
        self._rewards = []
        self._states = []
        self.policy_losses = deque(maxlen=500)
        self.value_losses = deque(maxlen=500)
        self.entropies = deque(maxlen=500)

    def _dist_and_value(self, obs, mask=None):
        state = torch.as_tensor(np.asarray(obs, dtype=np.float32).flatten(),
                                device=self.device).unsqueeze(0)
        logits, value = self.net(state)
        logits = logits.squeeze(0)
        if mask is not None:
            mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=self.device)
            logits = logits.masked_fill(~mask_t, -1e9)
        return torch.distributions.Categorical(logits=logits), value.squeeze(0)

    def get_action(self, obs, valid_actions=None):
        obs_raw, mask_list = self._unpack(obs)
        mask = mask_list
        if mask is None and valid_actions is not None:
            mask = [1 if i in valid_actions else 0 for i in range(self.act_dim)]
        dist, _ = self._dist_and_value(obs_raw, mask)
        # Evaluation (TRAIN=False) is greedy, matching the trainer\'s "greedy eval".
        action = dist.sample() if self.TRAIN else dist.probs.argmax(dim=-1)
        if self.TRAIN:
            self._log_probs.append(dist.log_prob(action))
            self._actions.append(int(action.item()))
            self._masks.append(
                np.asarray(mask, dtype=bool) if mask is not None else np.ones(self.act_dim, dtype=bool)
            )
            self._states.append(
                torch.as_tensor(np.asarray(obs_raw, dtype=np.float32).flatten(),
                                device=self.device)
            )
        return int(action.item())

    def update(self, obs, action, reward, next_obs, done):
        self._rewards.append(float(reward))

    def end_of_episode(self):
        if not self.TRAIN or not self._log_probs:
            self._reset()
            return
        # Monte-Carlo returns
        returns, G = [], 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        n = len(returns)
        states = torch.stack(self._states[:n])
        logits, values = self.net(states)
        masks = torch.as_tensor(np.stack(self._masks[:n]), device=self.device)
        logp_all = torch.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)
        # Log-probability of the action actually taken at each step. This used to
        # gather with torch.arange(n) - the timestep index, not the action - so
        # A2C reinforced arbitrary actions (always action 0 on 1-step episodes).
        actions_t = torch.as_tensor(self._actions[:n], dtype=torch.long, device=self.device)
        log_probs = logp_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        advantages = returns_t - values.detach()
        # Normalising a handful of advantages maps them to roughly ±1 whatever
        # their values, which discards the baseline. Only do it with enough steps.
        if advantages.numel() >= 8 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        entropy = -(logp_all.exp() * logp_all).sum(-1).sum()

        policy_loss = -(log_probs * advantages).sum()
        value_loss = nn.functional.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optim.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropies.append(float(entropy.item()) / max(len(returns), 1))
        self._reset()

    def _reset(self):
        self._log_probs = []
        self._rewards = []
        self._states = []
        self._actions = []
        self._masks = []

    def get_metrics(self):
        return {
            "policy_loss": float(np.mean(self.policy_losses)) if self.policy_losses else 0.0,
            "value_loss": float(np.mean(self.value_losses)) if self.value_losses else 0.0,
            "entropy": round(float(np.mean(self.entropies)), 4) if self.entropies else 0.0,
        }

    def save_model(self, path):
        torch.save(
            {
                "format": "a2c-v1",
                "agent_name": self.SPEC["name"],
                "net": self.net.state_dict(),
                "params": self.params,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_size": self.params["hidden_size"],
            },
            str(path),
        )

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["net"])
