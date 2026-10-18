"""Proximal Policy Optimization (PPO, clipped variant) for discrete actions.

Rollouts are collected in fixed-size chunks of environment steps; the policy is
then improved over several epochs of minibatch updates with a clipped
surrogate objective - the workhorse algorithm of modern deep RL.
"""

from collections import deque
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .base import Agent, ParamSpec, build_spec


class PPONet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.log_std = None  # discrete only

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


PARAMS = [
    ParamSpec("lr", "Learning rate", 3e-4, "float", "Adam learning rate", min=1e-6, max=1.0),
    ParamSpec("gamma", "Discount factor", 0.99, "float", "Reward discount", min=0.0, max=0.9999),
    ParamSpec("gae_lambda", "GAE lambda", 0.95, "float",
              "Generalised advantage estimation smoothing; 1.0 = high variance, 0 = TD",
              min=0.0, max=1.0),
    ParamSpec("clip_eps", "Clip epsilon", 0.2, "float",
              "PPO clip range - how far the new policy may move per update",
              min=0.01, max=0.5),
    ParamSpec("rollout_steps", "Rollout steps", 2048, "int",
              "Environment steps collected before each update round", min=64, max=65536,
              advanced=True),
    ParamSpec("epochs", "Update epochs", 10, "int",
              "Passes over the rollout data each update", min=1, max=100),
    ParamSpec("minibatch_size", "Minibatch size", 256, "int", "Samples per gradient step",
              min=16, max=4096, advanced=True),
    ParamSpec("entropy_coef", "Entropy coefficient", 0.01, "float",
              "Exploration bonus in the loss", min=0.0, max=1.0),
    ParamSpec("value_coef", "Value loss coef", 0.5, "float",
              "Weight of value-function loss", min=0.0, max=10.0, advanced=True),
    ParamSpec("max_grad_norm", "Grad clip", 0.5, "float", "Max gradient norm",
              min=0.05, max=100.0, advanced=True),
    ParamSpec("hidden_size", "Hidden units", 128, "int", "Width of hidden layers",
              min=8, max=4096, advanced=True),
]

_SPEC = build_spec(
    name="ppo",
    label="PPO",
    description="Clipped proximal policy optimisation: stable multi-epoch on-policy "
    "updates with GAE advantages. The default choice for most problems.",
    category="actor-critic",
    policy_type="on-policy",
    status="implemented",
    # pixel observations are flattened for the MLP; check_compat warns about it
    obs_types=["vector", "onehot", "image"],
    params=PARAMS,
    presets=[
        {"name": "beginner-default", "label": "Beginner default",
         "description": "Standard PPO hyperparameters for CartPole-class tasks.",
         "overrides": {}, "episodes": 300},
        {"name": "quick-demo", "label": "Quick demo",
         "description": "Small rollouts so updates (and learning) happen fast.",
         "overrides": {"rollout_steps": 512, "epochs": 6}, "episodes": 120},
        {"name": "stable-baseline", "label": "Stable baseline",
         "description": "Larger rollouts and conservative clipping.",
         "overrides": {"clip_eps": 0.15, "rollout_steps": 4096}, "episodes": 800},
        {"name": "fast-iteration", "label": "Fast iteration",
         "description": "Minimal rollouts to sanity-check setups.",
         "overrides": {"rollout_steps": 256}, "episodes": 80},
    ],
    beginner=(
        "PPO collects a batch of experience, then squeezes several careful improvement "
        "steps out of it. The famous 'clip' prevents any single update from changing the "
        "policy too much, which is what makes PPO famously stable and forgiving to tune."
    ),
    technical=(
        "Advantages via GAE(λ). The clipped surrogate L = E[min(r_t·Â_t, "
        "clip(r_t, 1±ε)·Â_t)] with probability ratio r_t = π_new/π_old. Value targets are "
        "GAE returns; total loss adds entropy bonus and value MSE. K epochs of shuffled "
        "minibatch Adam updates per rollout."
    ),
    intuition=(
        "Trust regions without second-order optimisation: improve confidently, but never "
        "move so far that your data stops describing your policy."
    ),
    update_rule=r"L^{CLIP} = Ê[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]",
    pseudocode=(
        "repeat:\n"
        "    collect `rollout_steps` steps with π_old\n"
        "    compute GAE(λ) advantages & returns\n"
        "    for epoch in 1..K:\n"
        "        for minibatch:\n"
        "            r = exp(logπ_new(a|s) - logπ_old(a|s))\n"
        "            L_pol = -min(r*A, clamp(r,1±eps)*A)\n"
        "            L_val = mse(V(s), G)\n"
        "            step on L_pol + c_v*L_val - c_e*H[π]"
    ),
    lifecycle=(
        "1. Interact for `rollout_steps` environment steps.\n"
        "2. Compute GAE advantages and returns.\n"
        "3. Run `epochs` passes of minibatch clipped-updates.\n"
        "4. Discard data and repeat."
    ),
    learns="A softmax policy + state-value function (separate heads on a tanh trunk).",
    exploration="Stochastic sampling + entropy bonus; no explicit ε schedule needed.",
    replay_buffer="Not used - on-policy; old rollouts are discarded after their update round.",
    target_network="Not used.",
    advantage="GAE(λ) with the learned critic.",
    networks="Tanh MLP trunk → linear policy head + linear value head.",
    failure_modes=[
        "Too-large lr or clip_eps → performance cliffs mid-training",
        "Value-function instability shows as reward crashes after good phases",
        "Very small rollout_steps gives noisy advantage estimates",
    ],
    expected_metrics=[
        {"key": "policy_loss", "meaning": "Clipped surrogate loss."},
        {"key": "value_loss", "meaning": "Critic MSE against GAE returns."},
        {"key": "entropy", "meaning": "Policy entropy per step - should decay gently."},
        {"key": "approx_kl", "meaning": "How far each update moved the policy - keep < ~0.02."},
    ],
    related=["a2c", "reinforce_baseline"],
)


class PPOAgent(Agent):
    SPEC = _SPEC

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.obs_dim = obs_dim
        dev = torch.device(device)
        self.net = PPONet(obs_dim, act_dim, params["hidden_size"]).to(dev)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=params["lr"])
        p = self.params
        self.gamma, self.lam = p["gamma"], p["gae_lambda"]
        self.clip_eps, self.epochs = p["clip_eps"], p["epochs"]
        self.minibatch = min(p["minibatch_size"], p["rollout_steps"])
        self.entropy_coef, self.value_coef = p["entropy_coef"], p["value_coef"]
        self.max_grad_norm = p["max_grad_norm"]

        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[int] = []
        self._logp_buf: List[float] = []
        self._rew_buf: List[float] = []
        self._done_buf: List[bool] = []
        self._val_buf: List[float] = []

        self.policy_losses = deque(maxlen=500)
        self.value_losses = deque(maxlen=500)
        self.entropies = deque(maxlen=2000)
        self.kls = deque(maxlen=2000)
        self._updates = 0

    # ------------------------------------------------------------------ #
    def _dist(self, logits, mask=None):
        if mask is not None:
            mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=self.device)
            logits = logits.masked_fill(~mask_t, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, obs, valid_actions=None):
        obs_raw, mask_list = self._unpack(obs)
        mask = mask_list
        if mask is None and valid_actions is not None:
            mask = [1 if i in valid_actions else 0 for i in range(self.act_dim)]
        state = torch.as_tensor(np.asarray(obs_raw, dtype=np.float32).flatten(),
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(state)
            dist = self._dist(logits.squeeze(0), mask)
            # Evaluation (TRAIN=False) is greedy, matching the trainer\'s "greedy eval".
            action = dist.sample() if self.TRAIN else dist.probs.argmax(dim=-1)
            logp = dist.log_prob(action).item()
            v = float(value.item())
        if self.TRAIN:
            self._obs_buf.append(np.asarray(obs_raw, dtype=np.float32).flatten())
            self._act_buf.append(int(action.item()))
            self._logp_buf.append(logp)
            self._val_buf.append(v)
            if not hasattr(self, "_mask_buf"):
                self._mask_buf = []
            self._mask_buf.append(
                np.asarray(mask, dtype=bool) if mask is not None
                else np.ones(self.act_dim, dtype=bool)
            )
        return int(action.item())

    def update(self, obs, action, reward, next_obs, done):
        if not self.TRAIN:
            return
        self._rew_buf.append(float(reward))
        self._done_buf.append(bool(done))
        if len(self._rew_buf) >= self.params["rollout_steps"]:
            self._update()

    def end_of_episode(self):
        # flush partial rollout at episode boundary only when training is ending;
        # normal operation waits for rollout_steps.
        pass

    def finalize_training(self):
        if self._rew_buf:
            self._update()

    # ------------------------------------------------------------------ #
    def _compute_gae(self):
        n = len(self._rew_buf)
        adv = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        next_value = 0.0  # rollouts end on done or are cut; bootstrap approximated by 0
        for t in reversed(range(n)):
            nonterminal = 0.0 if self._done_buf[t] else 1.0
            delta = self._rew_buf[t] + self.gamma * next_value * nonterminal - self._val_buf[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_value = self._val_buf[t]
        returns = adv + np.asarray(self._val_buf, dtype=np.float32)
        return adv, returns

    def _update(self):
        dev = torch.device(self.device)
        adv, returns = self._compute_gae()
        obs = np.asarray(self._obs_buf, dtype=np.float32)
        acts = np.asarray(self._act_buf, dtype=np.int64)
        old_logp = np.asarray(self._logp_buf, dtype=np.float32)

        obs_t = torch.as_tensor(obs, device=dev)
        acts_t = torch.as_tensor(acts, device=dev)
        old_logp_t = torch.as_tensor(old_logp, device=dev)
        adv_t = torch.as_tensor(adv, device=dev)
        ret_t = torch.as_tensor(returns, device=dev)

        # masks recorded during rollout collection
        masks = getattr(self, "_mask_buf", None)
        masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=dev) \
            if masks else None

        adv_norm = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = len(obs_t)
        idx = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch):
                mb = idx[start: start + self.minibatch]
                if len(mb) < 4:
                    continue
                logits, values = self.net(obs_t[mb])
                dist = self._dist(logits, masks_t[mb] if masks_t is not None else None)
                new_logp = dist.log_prob(acts_t[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logp - old_logp_t[mb]).exp()
                mb_adv = adv_norm[mb]
                pol1 = ratio * mb_adv
                pol2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(pol1, pol2).mean()
                value_loss = nn.functional.mse_loss(values, ret_t[mb])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                with torch.no_grad():
                    approx_kl = (old_logp_t[mb] - new_logp).mean().item()

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                self.policy_losses.append(float(policy_loss.item()))
                self.value_losses.append(float(value_loss.item()))
                self.entropies.append(float(entropy.item()))
                self.kls.append(approx_kl)
                self._updates += 1

        self._reset_buffers()

    def _reset_buffers(self):
        self._obs_buf, self._act_buf, self._logp_buf = [], [], []
        self._rew_buf, self._done_buf, self._val_buf = [], [], []
        self._mask_buf = []

    def get_metrics(self):
        return {
            "policy_loss": float(np.mean(self.policy_losses)) if self.policy_losses else 0.0,
            "value_loss": float(np.mean(self.value_losses)) if self.value_losses else 0.0,
            "entropy": round(float(np.mean(self.entropies)), 4) if self.entropies else 0.0,
            "approx_kl": round(float(np.mean(self.kls)), 6) if self.kls else 0.0,
            "updates": self._updates,
        }

    def save_model(self, path):
        torch.save(
            {
                "format": "ppo-v1",
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
