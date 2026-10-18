"""CrossQ: SAC without target networks, made stable by batch normalisation.

Bhatt et al. (ICLR 2024). Target networks slow learning down on purpose, so
the critic's targets move slowly. CrossQ removes them and instead
batch-normalises the critics, feeding current (s, a) and next (s', a') pairs
through the critic in one joint batch. The normalisation statistics then cover
both distributions, which keeps bootstrapping stable without a lagging copy.

Differences from the paper, for CPU scale: critics are 512 wide with standard
batch normalisation (the paper uses 2048 and batch renormalisation), and the
actor has no batch normalisation.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ParamSpec, build_spec
from .continuous import SACAgent


def _bn_q(in_dim: int, hidden: int, momentum: float) -> nn.Module:
    return nn.Sequential(
        nn.BatchNorm1d(in_dim, momentum=momentum),
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden, momentum=momentum), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden, momentum=momentum), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class BNTwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, momentum):
        super().__init__()
        self.q1 = _bn_q(obs_dim + act_dim, hidden, momentum)
        self.q2 = _bn_q(obs_dim + act_dim, hidden, momentum)

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def _params():
    out = []
    for d in SACAgent.SPEC["params"]:
        d = dict(d)
        if d["name"] == "tau":
            continue  # no target network to update
        if d["name"] == "hidden_size":
            d["default"] = 512
            d["help"] = "Width of the critics (the paper uses 2048; 512 suits a CPU)."
        if d["name"] in ("lr", "critic_lr"):
            d["default"] = 1e-3
        if d["name"] == "network":
            continue
        out.append(ParamSpec(**{k: d[k] for k in ("name", "label", "default", "type", "help",
                                                  "min", "max", "choices", "advanced",
                                                  "recommended") if k in d}))
    return out + [
        ParamSpec("policy_delay", "Policy delay", 3, "int",
                  "Critic updates per actor update (the paper uses 3).", min=1, max=20),
        ParamSpec("bn_momentum", "Batch-norm momentum", 0.01, "float",
                  "How quickly the running statistics used at evaluation adapt.",
                  min=0.0001, max=1.0, advanced=True),
        ParamSpec("adam_beta1", "Adam β1", 0.5, "float",
                  "CrossQ lowers Adam's momentum from 0.9 to 0.5.", min=0.0, max=0.999,
                  advanced=True),
    ]


class CrossQAgent(SACAgent):
    SPEC = build_spec(
        name="crossq", label="CrossQ", category="actor-critic", policy_type="off-policy",
        obs_types=["vector"], action_types=["continuous"],
        description="SAC without target networks: batch-normalised critics see current and "
        "next state-action pairs in one batch, which keeps bootstrapping stable.",
        params=_params(),
        presets=[{"name": "default", "label": "Default", "description": "Paper settings at CPU width.",
                  "overrides": {}, "episodes": 60}],
        beginner=(
            "SAC keeps a slow-moving copy of its critic so its learning targets do not jump "
            "around, which also slows learning. CrossQ removes the copy and keeps things stable "
            "a different way: it normalises the critic's inputs and features over each batch."
        ),
        technical=(
            "Critic: Q(s,a) with BatchNorm after the input and each hidden layer. The TD target "
            "uses the same critic, evaluated in one forward pass on the concatenation of "
            "(s, a) and (s', a'~π), so batch statistics are shared: y = r + γ(1−d)(min Q(s',a') "
            "− α log π(a'|s')), with gradients stopped through the target half. No target network, "
            "no Polyak averaging. Adam β1 = 0.5, actor updated every 3 critic steps."
        ),
        intuition="Stability from normalisation instead of from deliberately lagging targets.",
        update_rule="[Q(s,a), Q(s',a')] = Q_BN([s;s'], [a;a'])  ;  y = r + γ(1−d)(min Q(s',a') − α log π)",
        pseudocode=(
            "for each step:\n"
            "    a' ~ π(s')\n"
            "    q, q_next = critic(concat(s, s'), concat(a, a'))   # one batch-normalised pass\n"
            "    y = r + γ(1-done)(min(q_next).detach() - α log π(a'|s'))\n"
            "    critic loss = (q - y)^2\n"
            "    every 3 steps: actor and temperature as in SAC"
        ),
        learns="An actor, twin batch-normalised critics and the temperature α.",
        exploration="Stochastic policy with automatic entropy tuning, as in SAC.",
        replay_buffer="Uniform replay, as in SAC.",
        networks="Gaussian actor (MLP); twin critics with BatchNorm layers.",
        failure_modes=["Batch-norm statistics shift as the data changes; momentum matters",
                       "Slower per update than SAC because the critics are wider"],
        expected_metrics=[{"key": "critic_loss", "meaning": "TD error of the batch-normalised critics."},
                          {"key": "alpha", "meaning": "Entropy temperature."}],
        recommended_envs=["pendulum", "half_cheetah", "hopper", "walker2d", "mountain_car_continuous"],
        related=["sac", "td3"],
        paper="Bhatt et al., 2024. CrossQ: Batch Normalization in Deep RL for Greater Sample Efficiency and Simplicity.",
    )

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        params = {**params, "tau": 0.0}  # SAC's constructor reads tau; nothing uses it here
        super().__init__(obs_dim, act_dim, params, device)
        p = self.params
        self.critic = BNTwinCritic(obs_dim, act_dim, int(p["hidden_size"]), float(p["bn_momentum"])).to(self.dev)
        self.critic_target = self.critic  # no separate target; kept so save/load work unchanged
        betas = (float(p["adam_beta1"]), 0.999)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=p["critic_lr"], betas=betas)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=p["lr"], betas=betas)
        if self.auto_entropy:
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=p["alpha_lr"], betas=betas)

    def _learn(self):
        obs, act, rew, next_obs, done = self._sample_batch()
        b = obs.shape[0]
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            next_logp = next_logp.squeeze(-1)

        # one joint, batch-normalised pass over current and next pairs
        self.critic.train()
        q1_all, q2_all = self.critic(torch.cat([obs, next_obs]), torch.cat([act, next_action]))
        q1, q1_next = q1_all[:b], q1_all[b:]
        q2, q2_next = q2_all[:b], q2_all[b:]
        with torch.no_grad():
            y = rew + self.gamma * (1.0 - done) * (torch.min(q1_next, q2_next) - self.alpha * next_logp)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        self._critic_losses.append(critic_loss.item())
        self.updates += 1

        if self.updates % int(self.params["policy_delay"]) == 0:
            self.critic.eval()  # the actor is judged with running statistics, not batch ones
            new_action, logp = self.actor.sample(obs)
            logp = logp.squeeze(-1)
            q1_new, q2_new = self.critic(obs, new_action)
            actor_loss = (self.alpha * logp - torch.min(q1_new, q2_new)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optim.step()
            self.critic.train()
            if self.auto_entropy:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = float(self.log_alpha.exp().item())
            self._actor_losses.append(actor_loss.item())
            self._entropies.append(float(-logp.mean().item()))
            self._alphas.append(self.alpha)
