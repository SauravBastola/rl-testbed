"""World-model agents: learn how the environment works, then use the model.

Both agents share a learned model (:class:`EnsembleModel`): several small
networks, each predicting the change in state, the reward, and whether the
episode ends, from the current state and action. Disagreement between members
is a cheap measure of how unsure the model is.

- PETS plans at every step: it imagines many action sequences in the model and
  executes the first action of the best one (cross-entropy method), then
  replans. No policy network at all (Chua et al., 2018).
- Dreamer-lite never plans when acting: it trains an actor-critic purely on
  rollouts imagined by the model, the idea behind Dreamer (Hafner et al.),
  without the pixel encoder. Continuous actions learn by backpropagating
  returns through the model's dynamics; discrete ones by policy gradient.

Both report how fast the model's predictions drift from reality (error 1, 5
and 15 steps ahead) and publish an imagined-vs-real trajectory for the run page.
"""

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent, ParamSpec, build_spec

HORIZONS = (1, 5, 15)


def mlp(inp, out, hidden, layers=2):
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    return nn.Sequential(*mods, nn.Linear(d, out))


class EnsembleModel(nn.Module):
    """E members predicting (Δs, r, done-logit) from normalised (s, a).

    The members' weights are stacked so all of them run as one batched matrix
    multiply (``torch.baddbmm``): far faster than looping over member networks,
    which matters because planning evaluates the model thousands of times a step.
    """

    def __init__(self, obs_dim, act_in, hidden, members):
        super().__init__()
        self.obs_dim, self.E = obs_dim, members
        sizes = [obs_dim + act_in, hidden, hidden, obs_dim + 2]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = 1 / np.sqrt(fan_in)  # same initialisation as nn.Linear
            self.W.append(nn.Parameter(torch.empty(members, fan_in, fan_out).uniform_(-bound, bound)))
            self.b.append(nn.Parameter(torch.empty(members, 1, fan_out).uniform_(-bound, bound)))
        self.register_buffer("in_mu", torch.zeros(obs_dim + act_in))
        self.register_buffer("in_sd", torch.ones(obs_dim + act_in))
        self.register_buffer("d_mu", torch.zeros(obs_dim))
        self.register_buffer("d_sd", torch.ones(obs_dim))

    @property
    def members(self):  # kept for code that asks how many members there are
        return range(self.E)

    def fit_normaliser(self, s, a, ds):
        x = torch.cat([s, a], -1)
        self.in_mu.copy_(x.mean(0))
        self.in_sd.copy_(x.std(0).clamp_min(1e-3))
        self.d_mu.copy_(ds.mean(0))
        self.d_sd.copy_(ds.std(0).clamp_min(1e-3))

    def forward_all(self, s, a):
        """s, a: (E, N, ·) -> raw outputs (E, N, obs_dim + 2), member e on slice e."""
        h = (torch.cat([s, a], -1) - self.in_mu) / self.in_sd
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, w)
            if i < last:
                h = F.silu(h)
        return h

    def _decode(self, s, out):
        ds = out[..., : self.obs_dim] * self.d_sd + self.d_mu
        return s + ds, out[..., self.obs_dim], torch.sigmoid(out[..., self.obs_dim + 1])

    def step(self, e, s, a):
        """Next state, reward and done probability according to member e. s: (N, obs)."""
        h = (torch.cat([s, a], -1) - self.in_mu) / self.in_sd
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = torch.addmm(b[e], h, w[e])
            if i < last:
                h = F.silu(h)
        return self._decode(s, h)

    def step_particles(self, s, a):
        """Particle i follows member i mod E for the whole rollout (PETS TS-inf)."""
        n = s.shape[0]
        per = -(-n // self.E)  # ceil
        pad = per * self.E - n
        if pad:
            s = torch.cat([s, s[:pad]])
            a = torch.cat([a, a[:pad]])
        # interleaved assignment: particle i -> member i mod E
        s_e = s.view(per, self.E, -1).transpose(0, 1)
        a_e = a.view(per, self.E, -1).transpose(0, 1)
        nxt, r, d = self._decode(s_e, self.forward_all(s_e, a_e))
        back = lambda x: x.transpose(0, 1).reshape(per * self.E, *x.shape[2:])[:n]  # noqa: E731
        return back(nxt), back(r), back(d)


_MODEL_PARAMS = [
    ParamSpec("ensemble", "Ensemble members", 5, "int",
              "Independent dynamics networks; their disagreement measures uncertainty.",
              min=1, max=10),
    ParamSpec("hidden_size", "Hidden units", 200, "int", "Width of each member.", min=16, max=1024,
              advanced=True),
    ParamSpec("model_lr", "Model learning rate", 1e-3, "float", "Adam learning rate for the model.",
              min=1e-5, max=1e-1, advanced=True),
    ParamSpec("model_steps", "Model updates per episode", 200, "int",
              "Gradient steps on the model after each episode.", min=10, max=5000),
    ParamSpec("batch_size", "Batch size", 256, "int", "Transitions per model update.", min=16,
              max=4096, advanced=True),
    ParamSpec("warmup_episodes", "Random episodes", 2, "int",
              "Episodes of random actions before the model is trusted.", min=0, max=50),
    ParamSpec("horizon", "Imagination horizon", 15, "int",
              "Steps imagined ahead.", min=2, max=100),
    ParamSpec("gamma", "Discount", 0.99, "float", "Discount for imagined returns.", min=0.0,
              max=0.9999),
]


class _WorldModelAgent(Agent):
    """Shared: replay of real transitions, the ensemble, its training and diagnostics."""

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(act_dim, params, device)
        p = self.params
        self.obs_dim = obs_dim
        self.continuous = False  # set by set_action_bounds for continuous envs
        self.act_in = act_dim
        self.model = EnsembleModel(obs_dim, act_dim, p["hidden_size"], p["ensemble"])
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=p["model_lr"])
        self.S: List[np.ndarray] = []
        self.A: List[np.ndarray] = []
        self.R: List[float] = []
        self.S2: List[np.ndarray] = []
        self.D: List[float] = []
        self.episode = 0
        self._ep_start = 0
        self.model_losses = deque(maxlen=50)
        self.horizon_err: Dict[int, float] = {}
        self._imagination: Optional[Dict] = None
        self.last_step_truncated = False

    # -------------------------------------------------------------- actions
    def set_action_bounds(self, low, high):
        super().set_action_bounds(low, high)
        self.continuous = True

    def _encode_action(self, action) -> np.ndarray:
        """Model input: normalised [-1, 1] vector (continuous) or one-hot (discrete)."""
        if self.continuous:
            a = np.asarray(action, dtype=np.float32).flatten()
            return 2 * (a - self.act_low) / (self.act_high - self.act_low + 1e-8) - 1
        one = np.zeros(self.act_dim, dtype=np.float32)
        one[int(action)] = 1.0
        return one

    def _env_action(self, a_norm):
        """From the model's action space to the environment's."""
        if self.continuous:
            a = np.clip(np.asarray(a_norm, dtype=np.float32), -1, 1)
            return self.act_low + (a + 1) * 0.5 * (self.act_high - self.act_low)
        return int(a_norm)

    def _random_action(self, mask=None):
        if self.continuous:
            return np.random.uniform(self.act_low, self.act_high).astype(np.float32)
        valid = [i for i in range(self.act_dim) if mask is None or mask[i]]
        return int(np.random.choice(valid))

    def _state(self, obs):
        raw, mask = self._unpack(obs)
        return np.asarray(raw, dtype=np.float32).flatten(), mask

    @property
    def warming_up(self):
        return self.episode < self.params["warmup_episodes"] or len(self.S) < 64

    # -------------------------------------------------------------- learning
    def update(self, obs, action, reward, next_obs, done):
        if not self.TRAIN:
            return
        s, _ = self._state(obs)
        s2, _ = self._state(next_obs)
        self.S.append(s)
        self.A.append(self._encode_action(action))
        self.R.append(float(reward))
        self.S2.append(s2)
        # A time-limit truncation is not the world ending.
        self.D.append(float(done and not getattr(self, "last_step_truncated", False)))

    def _tensors(self):
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        return t(self.S), t(self.A), t(self.R), t(self.S2), t(self.D)

    def train_model(self):
        s, a, r, s2, d = self._tensors()
        n = len(s)
        self.model.fit_normaliser(s, a, s2 - s)
        p = self.params
        batch, e_count = min(p["batch_size"], n), self.model.E
        for _ in range(p["model_steps"]):
            idx = torch.randint(0, n, (e_count, batch))  # a bootstrap sample per member
            out = self.model.forward_all(s[idx], a[idx])
            target_ds = ((s2[idx] - s[idx]) - self.model.d_mu) / self.model.d_sd
            loss = (F.mse_loss(out[..., : self.obs_dim], target_ds)
                    + F.mse_loss(out[..., self.obs_dim], r[idx])
                    + F.binary_cross_entropy_with_logits(out[..., self.obs_dim + 1], d[idx]))
            self.model_opt.zero_grad()
            loss.backward()
            self.model_opt.step()
            self.model_losses.append(loss.item())

    @torch.no_grad()
    def _horizon_diagnostics(self, start, end):
        """Open-loop: feed the real actions to the model from the real start state."""
        if end - start < 2:
            return
        s, a, _, s2, _ = self._tensors()
        s, a, s2 = s[start:end], a[start:end], s2[start:end]
        scale = self.model.in_sd[: self.obs_dim]
        pred = s[0:1].repeat(len(self.model.members), 1)
        imagined, errs = [], {}
        for t in range(len(s)):
            nxt = torch.stack([self.model.step(e, pred[e:e + 1], a[t:t + 1])[0][0]
                               for e in range(len(self.model.members))])
            pred = nxt
            mean = pred.mean(0)
            imagined.append(mean.tolist())
            err = (((mean - s2[t]) / scale) ** 2).mean().sqrt().item()
            if t + 1 in HORIZONS:
                errs[t + 1] = err
        self.horizon_err.update(errs)
        n = min(len(s), 60)
        self._imagination = {
            "episode": self.episode, "steps": n,
            "real": s2[:n].tolist(), "imagined": imagined[:n],
            "note": "Open-loop prediction: the model is given the real start state and the real "
                    "actions, and never sees the real states again.",
        }

    def end_of_episode(self):
        if not self.TRAIN:
            return
        start, end = self._ep_start, len(self.S)
        self.episode += 1
        if len(self.S) >= 64:
            self._horizon_diagnostics(start, end)  # before training: honest test on new data
            self.train_model()
            self.after_model_update()
        self._ep_start = len(self.S)

    def after_model_update(self):
        """Hook for agents that also learn a policy from the model."""

    def artifacts(self):
        return {"imagination.json": self._imagination} if self._imagination else {}

    def get_metrics(self):
        m = {"model_loss": round(float(np.mean(self.model_losses)), 5) if self.model_losses else 0.0,
             "transitions": len(self.S)}
        for h, v in self.horizon_err.items():
            m[f"model_error_{h}"] = round(v, 4)
        return m

    def save_model(self, path):
        torch.save({"format": "world-model-v1", "agent_name": self.SPEC["name"],
                    "model": self.model.state_dict(), "extra": self._extra_state(),
                    "params": self.params, "continuous": self.continuous,
                    "act_low": getattr(self, "act_low", None), "act_high": getattr(self, "act_high", None)},
                   str(path))

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self._load_extra(ckpt.get("extra") or {})
        self.episode = max(self.episode, self.params["warmup_episodes"])
        self.S = [np.zeros(self.obs_dim, dtype=np.float32)] * 64  # a loaded model is trusted

    def _extra_state(self):
        return {}

    def _load_extra(self, state):
        pass


_SHARED_DOCS = dict(
    obs_types=["vector"], action_types=["discrete", "continuous"],
    failure_modes=[
        "Model errors compound over long horizons: plans that look great in the model fail for real",
        "Learns slowly where the dynamics are complex or high-dimensional",
        "Planning at every step is far slower per step than a policy network",
    ],
    recommended_envs=["pendulum", "cartpole", "mountain_car_continuous"],
    poor_envs=["humanoid"],
)
_MODEL_METRICS = [
    {"key": "model_loss", "meaning": "Training loss of the learned model."},
    {"key": "model_error_1", "meaning": "Prediction error one step ahead on the latest episode (normalised RMSE)."},
    {"key": "model_error_15", "meaning": "Error 15 steps ahead: how fast imagination drifts from reality."},
]


class PETSAgent(_WorldModelAgent):
    SPEC = build_spec(
        name="pets", label="PETS (planning with a learned model)", category="model-based",
        policy_type="off-policy",
        description="Learn an ensemble model of the environment, then plan at every step by "
        "imagining action sequences in it and taking the first action of the best.",
        params=_MODEL_PARAMS + [
            ParamSpec("population", "Candidate plans", 300, "int",
                      "Action sequences imagined per planning iteration.", min=10, max=5000),
            ParamSpec("cem_iters", "CEM iterations", 3, "int",
                      "Rounds of refining the candidate distribution.", min=1, max=20),
            ParamSpec("elite_frac", "Elite fraction", 0.1, "float",
                      "Share of best plans the next round is fitted to.", min=0.01, max=0.5,
                      advanced=True),
        ],
        presets=[{"name": "default", "label": "Default", "description": "Pendulum and CartPole.",
                  "overrides": {}, "episodes": 15}],
        beginner=(
            "First, learn how the world responds to actions: watch what happens and fit a model. "
            "Then, before every move, try hundreds of possible futures in your head, and do the "
            "first step of the best one. Next step, think again."
        ),
        technical=(
            "An ensemble of E networks predicts (Δs, r, done) from (s, a), each on a bootstrap of "
            "the data. Planning uses the cross-entropy method over H-step action sequences: sample "
            "a population, score each by its imagined discounted return (each particle follows one "
            "member: trajectory sampling), refit the sampling distribution to the elite, repeat. "
            "Execute the first action, replan next step (model-predictive control)."
        ),
        intuition="Think before you act, using what you have learned about cause and effect.",
        update_rule="a_t = first action of argmax_{a_{t:t+H}} Σ_k γ^k r̂_k under the learned model",
        pseudocode=(
            "repeat each episode:\n"
            "    for each step:\n"
            "        dist = N(mean, std) over H-step action sequences\n"
            "        repeat K times:\n"
            "            plans = sample(dist);  score = imagined return of each plan\n"
            "            dist = fit(top plans by score)\n"
            "        act(first action of best plan)\n"
            "    train the ensemble on all real transitions"
        ),
        learns="A model of dynamics, reward and termination; no policy or value function.",
        exploration="Random actions for the first episodes, then the variety of candidate plans.",
        replay_buffer="Every real transition, used to train the model.",
        networks="An ensemble of MLPs (state, action) -> (state change, reward, done).",
        expected_metrics=_MODEL_METRICS + [
            {"key": "plan_return", "meaning": "Imagined return of the chosen plan."}],
        related=["dreamer_lite", "dyna_q", "sac"],
        paper="Chua et al., 2018. Deep RL in a Handful of Trials using Probabilistic Dynamics Models.",
        **_SHARED_DOCS,
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        self.plan_returns = deque(maxlen=200)
        self._mean = None  # warm start: last plan shifted by one step

    @torch.no_grad()
    def _score(self, s0, plans):
        """Imagined discounted return of each plan (P, H, act_in) from state s0."""
        p = self.params
        n = plans.shape[0]
        s = torch.as_tensor(s0, dtype=torch.float32).repeat(n, 1)
        total, alive, disc = torch.zeros(n), torch.ones(n), 1.0
        for t in range(plans.shape[1]):
            s, r, done = self.model.step_particles(s, plans[:, t])
            total += disc * alive * r
            alive = alive * (1 - done)
            disc *= p["gamma"]
        return total

    @torch.no_grad()
    def _plan(self, s0, mask=None):
        p = self.params
        h, n = p["horizon"], p["population"]
        n_elite = max(2, int(n * p["elite_frac"]))
        if self.continuous:
            mean = torch.zeros(h, self.act_in) if self._mean is None else self._mean
            std = torch.full((h, self.act_in), 0.5)
            for _ in range(p["cem_iters"]):
                plans = (mean + std * torch.randn(n, h, self.act_in)).clamp(-1, 1)
                score = self._score(s0, plans)
                elite = plans[score.topk(n_elite).indices]
                mean, std = elite.mean(0), elite.std(0).clamp_min(0.05)
            best = plans[score.argmax()]
            self._mean = torch.cat([mean[1:], torch.zeros(1, self.act_in)])  # shift for next step
            self.plan_returns.append(score.max().item())
            return best[0].numpy()
        logits = torch.zeros(h, self.act_dim)
        if mask is not None:
            logits[:, [i for i in range(self.act_dim) if not mask[i]]] = -1e9
        for _ in range(p["cem_iters"]):
            idx = torch.distributions.Categorical(logits=logits).sample((n,))  # (n, h)
            plans = F.one_hot(idx, self.act_dim).float()
            score = self._score(s0, plans)
            elite = idx[score.topk(n_elite).indices]
            freq = F.one_hot(elite, self.act_dim).float().mean(0) + 1e-3
            logits = freq.log()
        self.plan_returns.append(score.max().item())
        return int(idx[score.argmax(), 0])

    def get_action(self, obs, valid_actions=None):
        s, mask = self._state(obs)
        if self.TRAIN and self.warming_up:
            return self._random_action(mask)
        return self._env_action(self._plan(s, mask))

    def end_of_episode(self):
        super().end_of_episode()
        self._mean = None

    def get_metrics(self):
        m = super().get_metrics()
        if self.plan_returns:
            m["plan_return"] = round(float(np.mean(self.plan_returns)), 3)
        return m


class DreamerLiteAgent(_WorldModelAgent):
    SPEC = build_spec(
        name="dreamer_lite", label="Dreamer-lite (learning in imagination)", category="model-based",
        policy_type="off-policy",
        description="Learn a model of the environment, then train an actor-critic entirely on "
        "rollouts imagined by the model. Acting is a single policy forward pass.",
        params=_MODEL_PARAMS + [
            ParamSpec("ac_steps", "Actor-critic updates per episode", 100, "int",
                      "Imagination training steps after each episode.", min=1, max=5000),
            ParamSpec("ac_lr", "Actor-critic learning rate", 3e-4, "float",
                      "Adam learning rate for actor and critic.", min=1e-6, max=1e-1, advanced=True),
            ParamSpec("lambda_", "λ for returns", 0.95, "float",
                      "Mixes n-step returns in imagination, as in TD(λ).", min=0.0, max=1.0,
                      advanced=True),
            ParamSpec("entropy_coef", "Entropy bonus", 1e-3, "float",
                      "Keeps the imagined policy exploring.", min=0.0, max=1.0),
        ],
        presets=[{"name": "default", "label": "Default", "description": "Pendulum and CartPole.",
                  "overrides": {}, "episodes": 40}],
        beginner=(
            "Learn how the world works, then practise in your imagination: start from situations "
            "you have really seen, dream how they would continue under your policy, and improve "
            "the policy using those dreams. Real experience is only used to keep the dreams honest."
        ),
        technical=(
            "From real start states, roll the learned model forward H steps under the actor. "
            "The critic regresses λ-returns of imagined rewards; the actor maximises them, by "
            "backpropagating through the differentiable model for continuous actions (dynamics "
            "backpropagation) or by REINFORCE with the critic as baseline for discrete ones. "
            "Dreamer adds a learned latent space for pixels; this version works on state vectors."
        ),
        intuition="Practise in your head, then check against reality.",
        update_rule="V_λ(s_t) = r̂_t + γ[(1−λ)V(ŝ_{t+1}) + λV_λ(ŝ_{t+1})];  actor ↑ V_λ through the model",
        pseudocode=(
            "repeat each episode:\n"
            "    act with the actor; store real transitions\n"
            "    train the model on real data\n"
            "    repeat N times:\n"
            "        s = real states;  imagine H steps with actor and model\n"
            "        critic -> λ-returns;  actor -> maximise λ-returns"
        ),
        learns="A model, an actor and a critic. The actor never trains on real transitions.",
        exploration="A stochastic actor with an entropy bonus.",
        replay_buffer="Real transitions train the model and supply imagination start states.",
        networks="Model ensemble; actor and critic MLPs.",
        expected_metrics=_MODEL_METRICS + [
            {"key": "imagined_return", "meaning": "Mean λ-return in imagination; compare with real returns."},
            {"key": "actor_entropy", "meaning": "Entropy of the policy."}],
        related=["pets", "sac", "a2c"],
        paper="Hafner et al., Dream to Control (2020) and Mastering Diverse Domains through World Models (2023).",
        **_SHARED_DOCS,
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        h = self.params["hidden_size"]
        self._ac_ready = False
        self.critic = mlp(obs_dim, 1, h)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.params["ac_lr"])
        self.imagined = deque(maxlen=50)
        self.entropies = deque(maxlen=50)

    def _build_actor(self):
        out = 2 * self.act_dim if self.continuous else self.act_dim
        self.actor = mlp(self.obs_dim, out, self.params["hidden_size"])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.params["ac_lr"])
        self._ac_ready = True

    def _norm(self, s):
        return (s - self.model.in_mu[: self.obs_dim]) / self.model.in_sd[: self.obs_dim]

    def _dist(self, s):
        out = self.actor(self._norm(s))
        if self.continuous:
            mu, log_std = out.chunk(2, -1)
            return torch.distributions.Normal(mu, log_std.clamp(-5, 1).exp())
        return torch.distributions.Categorical(logits=out)

    def get_action(self, obs, valid_actions=None):
        s, mask = self._state(obs)
        if not self._ac_ready:
            self._build_actor()
        if self.TRAIN and self.warming_up:
            return self._random_action(mask)
        with torch.no_grad():
            d = self._dist(torch.as_tensor(s).unsqueeze(0))
            if self.continuous:
                a = torch.tanh(d.sample() if self.TRAIN else d.mean)[0].numpy()
                return self._env_action(a)
            return int((d.sample() if self.TRAIN else d.probs.argmax(-1))[0])

    def after_model_update(self):
        if not self._ac_ready:
            self._build_actor()
        p = self.params
        gamma, lam, horizon = p["gamma"], p["lambda_"], p["horizon"]
        s_all, *_ = self._tensors()
        self.model.requires_grad_(False)  # the model is fixed while the actor learns from it
        for _ in range(p["ac_steps"]):
            s = s_all[torch.randint(0, len(s_all), (min(256, len(s_all)),))]
            states, rewards, conts, logps, ents = [], [], [], [], []
            for _t in range(horizon):
                states.append(s)
                d = self._dist(s)
                if self.continuous:
                    a = torch.tanh(d.rsample())  # reparameterised: gradients flow through the model
                    ents.append(d.entropy().sum(-1))
                else:
                    idx = d.sample()
                    a = F.one_hot(idx, self.act_dim).float()
                    logps.append(d.log_prob(idx))
                    ents.append(d.entropy())
                member = np.random.randint(len(self.model.members))
                s, r, done = self.model.step(member, s, a)
                rewards.append(r)
                conts.append(1.0 - done)
            values = [self.critic(self._norm(x)).squeeze(-1) for x in states]
            bootstrap = self.critic(self._norm(s)).squeeze(-1)
            # λ-returns, backwards:  G_t = r_t + γ c_t [(1-λ) V(s_{t+1}) + λ G_{t+1}]
            g, returns = bootstrap, []
            for t in reversed(range(horizon)):
                v_next = values[t + 1] if t + 1 < horizon else bootstrap
                g = rewards[t] + gamma * conts[t] * ((1 - lam) * v_next + lam * g)
                returns.insert(0, g)
            R = torch.stack(returns)  # (H, B)
            with torch.no_grad():  # weight each step by the chance of still being alive
                alive = torch.cumprod(torch.stack([torch.ones_like(conts[0])] + conts[:-1]), 0)
            ent = torch.stack(ents)
            if self.continuous:
                actor_loss = -(alive * R).mean() - p["entropy_coef"] * (alive * ent).mean()
            else:
                adv = (R - torch.stack(values)).detach()
                actor_loss = -(alive * torch.stack(logps) * adv).mean() - p["entropy_coef"] * (alive * ent).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
            self.actor_opt.step()

            # critic: regress λ-returns on the (detached) imagined states
            v = torch.stack([self.critic(self._norm(x.detach())).squeeze(-1) for x in states])
            critic_loss = (alive * (v - R.detach()) ** 2).mean()
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
            self.critic_opt.step()

            self.imagined.append(R[0].mean().item())
            self.entropies.append(ent.mean().item())
        self.model.requires_grad_(True)

    def _extra_state(self):
        if not self._ac_ready:
            return {}
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def _load_extra(self, state):
        if state.get("actor"):
            if not self._ac_ready:
                self._build_actor()
            self.actor.load_state_dict(state["actor"])
            self.critic.load_state_dict(state["critic"])

    def get_metrics(self):
        m = super().get_metrics()
        if self.imagined:
            m["imagined_return"] = round(float(np.mean(self.imagined)), 3)
            m["actor_entropy"] = round(float(np.mean(self.entropies)), 4)
        return m
