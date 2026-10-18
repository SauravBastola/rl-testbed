"""More tabular and bandit methods.

Each of these is one idea away from Q-learning, which is exactly why they are
worth having side by side: Expected SARSA (average instead of max), Double
Q-learning (two tables against overestimation), Dyna-Q (learn a model and plan
with it), Monte Carlo control (no bootstrapping at all), UCB and Thompson
sampling (exploration by uncertainty instead of by coin flips).
"""

from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np

from .base import ParamSpec, build_spec
from .tabular import _COMMON_PARAMS, _TabularBase, _state_key

_TABULAR_PRESETS = [
    {"name": "beginner-default", "label": "Beginner default",
     "description": "Settings that solve Gridworld and CliffWalking.",
     "overrides": {}, "episodes": 300},
    {"name": "quick-demo", "label": "Quick demo",
     "description": "Fewer episodes with a faster step size.",
     "overrides": {"alpha": 0.2}, "episodes": 150},
]

_TABULAR_METRICS = [
    {"key": "epsilon", "meaning": "Exploration rate, decaying per episode."},
    {"key": "table_size", "meaning": "Distinct states visited so far."},
    {"key": "td_error", "meaning": "Recent mean absolute TD error; should fall."},
]


def _greedy(q, actions):
    return int(max(actions, key=lambda a: q[a]))


# --------------------------------------------------------------------------- #
class ExpectedSarsaAgent(_TabularBase):
    SPEC = build_spec(
        name="expected_sarsa",
        label="Expected SARSA",
        description="SARSA with the next action averaged out: the target uses the "
        "expected value under the epsilon-greedy policy, removing a source of variance.",
        category="value-based", policy_type="on-policy",
        params=_COMMON_PARAMS, presets=_TABULAR_PRESETS,
        beginner=(
            "SARSA learns from the one next action it happens to take. Expected SARSA "
            "instead asks: on average, over every action my policy might take next, "
            "what is the next state worth? Same idea, less luck in each update."
        ),
        technical=(
            "Target y = r + γ Σ_a' π(a'|s') Q(s',a') with π epsilon-greedy over Q. "
            "With ε→0 it becomes Q-learning; with stochastic π it is an on-policy "
            "method with lower variance than SARSA (van Seijen et al., 2009)."
        ),
        intuition="Instead of betting on one next move, it values the whole menu of "
        "next moves weighted by how likely it is to pick each.",
        update_rule="Q(s,a) ← Q(s,a) + α [ r + γ Σ_{a'} π(a'|s′) Q(s′,a′) − Q(s,a) ]",
        pseudocode=(
            "for each step:\n"
            "    a = eps-greedy(Q[s]);  s2, r, done = env.step(a)\n"
            "    pi = eps-greedy probabilities over Q[s2]\n"
            "    Q[s,a] += alpha*(r + gamma*sum(pi*Q[s2]) - Q[s,a])"
        ),
        learns="A state-action value table for the epsilon-greedy policy.",
        exploration="Linear epsilon-greedy decay per episode.",
        networks="None - lookup table.",
        failure_modes=[
            "Still needs a small, discrete state space",
            "Learns the value of the exploring policy, so it is cautious while ε is high",
        ],
        expected_metrics=_TABULAR_METRICS,
        recommended_envs=["cliff_walking", "gridworld", "windy_gridworld"],
        related=["sarsa", "q_learning"],
        paper="van Seijen et al., 2009. A theoretical and empirical analysis of Expected Sarsa.",
    )

    def update(self, obs, action, reward, next_obs, done):
        s, s2 = _state_key(obs), _state_key(next_obs)
        _, mask = self._unpack(next_obs)
        valid = self._mask_or_range(mask)
        target = float(reward)
        if not done:
            q2 = self.Q[s2]
            probs = np.full(len(valid), self.epsilon / len(valid))
            probs[valid.index(_greedy(q2, valid))] += 1.0 - self.epsilon
            target += self.gamma * float(np.dot(probs, [q2[a] for a in valid]))
        td = target - self.Q[s][int(action)]
        self.Q[s][int(action)] += self.alpha * td
        self.td_errors.append(abs(td))


# --------------------------------------------------------------------------- #
class DoubleQLearningAgent(_TabularBase):
    SPEC = build_spec(
        name="double_q",
        label="Double Q-learning",
        description="Two Q-tables: one picks the best next action, the other scores it. "
        "Removes the maximisation bias that makes Q-learning over-optimistic.",
        category="value-based", policy_type="off-policy",
        params=_COMMON_PARAMS, presets=_TABULAR_PRESETS,
        beginner=(
            "Taking the maximum of noisy estimates is biased upward: the action that "
            "looks best is often just the luckiest. Double Q-learning keeps two "
            "independent tables and lets one check the other's homework."
        ),
        technical=(
            "Maintains Q_A and Q_B. Each step a coin flip picks which to update; "
            "for Q_A the target is r + γ Q_B(s', argmax_a Q_A(s',a)). Actions are "
            "chosen epsilon-greedily on Q_A + Q_B (van Hasselt, 2010)."
        ),
        intuition="Selection and evaluation are done by different judges, so one "
        "judge's lucky guess is not automatically believed.",
        update_rule="Q_A(s,a) ← Q_A(s,a) + α [ r + γ Q_B(s′, argmax_{a'} Q_A(s′,a′)) − Q_A(s,a) ]",
        pseudocode=(
            "for each step:\n"
            "    a = eps-greedy(QA[s] + QB[s]);  s2, r, done = env.step(a)\n"
            "    if coin(): a* = argmax QA[s2];  QA[s,a] += alpha*(r + gamma*QB[s2,a*] - QA[s,a])\n"
            "    else:      a* = argmax QB[s2];  QB[s,a] += alpha*(r + gamma*QA[s2,a*] - QB[s,a])"
        ),
        learns="Two independent state-action value tables.",
        exploration="Linear epsilon-greedy on the sum of both tables.",
        networks="None - two lookup tables.",
        failure_modes=[
            "Each table sees half the updates, so learning is slower early on",
            "Can underestimate in environments where optimism helps exploration",
        ],
        expected_metrics=_TABULAR_METRICS,
        recommended_envs=["multi_armed_bandit", "frozen_lake", "gridworld"],
        related=["q_learning", "dqn_double"],
        paper="van Hasselt, 2010. Double Q-learning. NeurIPS.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        self.QB: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.act_dim))

    def get_action(self, obs, valid_actions=None):
        _, mask = self._unpack(obs)
        actions = self._mask_or_range(mask if mask is not None else valid_actions)
        if len(actions) > 1 and self.TRAIN and np.random.random() < self.epsilon:
            return int(np.random.choice(actions))
        s = _state_key(obs)
        return _greedy(self.Q[s] + self.QB[s], actions)

    def update(self, obs, action, reward, next_obs, done):
        s, s2 = _state_key(obs), _state_key(next_obs)
        _, mask = self._unpack(next_obs)
        valid = self._mask_or_range(mask)
        upd, other = (self.Q, self.QB) if np.random.random() < 0.5 else (self.QB, self.Q)
        target = float(reward)
        if not done:
            target += self.gamma * other[s2][_greedy(upd[s2], valid)]
        td = target - upd[s][int(action)]
        upd[s][int(action)] += self.alpha * td
        self.td_errors.append(abs(td))

    def save_model(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"format": "double_q", "agent_name": "double_q", "Q": dict(self.Q),
                         "QB": dict(self.QB), "params": self.params,
                         "act_dim": self.act_dim, "episode_count": self._episode_count}, f)

    def load_model(self, path):
        import pickle

        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.act_dim), ckpt.get("Q", {}))
        self.QB = defaultdict(lambda: np.zeros(self.act_dim), ckpt.get("QB", {}))
        self._episode_count = ckpt.get("episode_count", 0)


# --------------------------------------------------------------------------- #
class DynaQAgent(_TabularBase):
    SPEC = build_spec(
        name="dyna_q",
        label="Dyna-Q",
        description="Q-learning plus a learned model of the world. After every real "
        "step it replays imagined transitions from the model to plan ahead.",
        category="model-based", policy_type="off-policy",
        params=_COMMON_PARAMS + [
            ParamSpec("planning_steps", "Planning steps", 10, "int",
                      "Simulated updates from the learned model per real step. "
                      "0 turns Dyna-Q back into Q-learning.", min=0, max=500),
        ],
        presets=[
            {"name": "beginner-default", "label": "Beginner default",
             "description": "Ten planning steps per real step.",
             "overrides": {}, "episodes": 50},
            {"name": "heavy-planning", "label": "Heavy planning",
             "description": "Fifty planning steps; very few real episodes needed.",
             "overrides": {"planning_steps": 50}, "episodes": 30},
            {"name": "no-planning", "label": "No planning (ablation)",
             "description": "Equivalent to Q-learning, for comparison.",
             "overrides": {"planning_steps": 0}, "episodes": 50},
        ],
        beginner=(
            "Every real experience is written into a notebook: 'in state s, doing a "
            "gave reward r and led to s′'. Between real steps the agent flips through "
            "the notebook and practises on remembered experiences. In a deterministic "
            "world this is like getting many free episodes per real one."
        ),
        technical=(
            "Direct RL: one-step Q-learning on each real transition. Model learning: "
            "a table Model(s,a) = (r, s′, done) (last observed outcome). Planning: n "
            "times per step, sample a previously seen (s,a), query the model and apply "
            "the same Q-learning update (Sutton, 1990)."
        ),
        intuition="Learning from imagination: experience is reused until its "
        "consequences have propagated through the value table.",
        update_rule="real: Q(s,a) ← Q(s,a) + α δ ;  then n × [ (s̃,ã) ~ seen, (r̃,s̃′) = M(s̃,ã), Q(s̃,ã) ← Q(s̃,ã) + α δ̃ ]",
        pseudocode=(
            "for each step:\n"
            "    a = eps-greedy(Q[s]);  s2, r, done = env.step(a)\n"
            "    Q[s,a] += alpha*(r + gamma*max(Q[s2]) - Q[s,a])\n"
            "    Model[s,a] = (r, s2, done)\n"
            "    repeat n times:\n"
            "        (ps, pa) = random previously seen pair\n"
            "        pr, ps2, pdone = Model[ps, pa]\n"
            "        Q[ps,pa] += alpha*(pr + gamma*max(Q[ps2])*(1-pdone) - Q[ps,pa])"
        ),
        lifecycle="Act, learn from the real step, record it in the model, then plan.",
        learns="A Q-table plus a deterministic tabular model of transitions and rewards.",
        exploration="Linear epsilon-greedy decay per episode.",
        replay_buffer="The model doubles as a de-duplicated replay memory of (s,a) pairs.",
        networks="None - lookup tables.",
        failure_modes=[
            "Assumes a deterministic world: stochastic transitions are remembered by "
            "their most recent outcome only",
            "If the world changes, a stale model keeps teaching outdated values",
            "Planning cost grows linearly with planning_steps",
        ],
        expected_metrics=_TABULAR_METRICS + [
            {"key": "model_size", "meaning": "State-action pairs stored in the model."},
        ],
        recommended_envs=["dyna_maze", "gridworld", "windy_gridworld"],
        related=["q_learning"],
        paper="Sutton, 1990. Integrated architectures for learning, planning, and reacting.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        self.model: Dict[tuple, tuple] = {}
        self._keys: list = []
        self.n_plan = int(self.params.get("planning_steps", 10))

    def _q_update(self, s, a, r, s2, done, valid=None):
        target = float(r)
        if not done:
            q2 = self.Q[s2]
            target += self.gamma * (max(q2[x] for x in valid) if valid else float(np.max(q2)))
        td = target - self.Q[s][a]
        self.Q[s][a] += self.alpha * td
        return td

    def update(self, obs, action, reward, next_obs, done):
        s, s2, a = _state_key(obs), _state_key(next_obs), int(action)
        _, mask = self._unpack(next_obs)
        td = self._q_update(s, a, reward, s2, done, self._mask_or_range(mask))
        self.td_errors.append(abs(td))
        if (s, a) not in self.model:
            self._keys.append((s, a))
        self.model[(s, a)] = (float(reward), s2, bool(done))
        if not self.TRAIN:
            return
        for _ in range(self.n_plan):
            ps, pa = self._keys[np.random.randint(len(self._keys))]
            pr, ps2, pdone = self.model[(ps, pa)]
            self._q_update(ps, pa, pr, ps2, pdone)

    def get_metrics(self):
        m = super().get_metrics()
        m["model_size"] = len(self.model)
        return m


# --------------------------------------------------------------------------- #
class MonteCarloControlAgent(_TabularBase):
    SPEC = build_spec(
        name="mc_control",
        label="Monte Carlo control",
        description="Waits until the episode ends, then moves each visited value toward "
        "the return that actually followed. No bootstrapping, no bias, high variance.",
        category="value-based", policy_type="on-policy",
        params=_COMMON_PARAMS,
        presets=_TABULAR_PRESETS,
        beginner=(
            "Temporal-difference methods guess the value of the next state. Monte Carlo "
            "never guesses: it plays the whole episode, adds up what really happened, "
            "and only then updates. Honest but slow when episodes are long."
        ),
        technical=(
            "Every-visit, constant-α Monte Carlo control with an ε-greedy policy. "
            "Returns G_t = Σ_k γ^k r_{t+k+1} are computed backwards at episode end and "
            "Q(s_t,a_t) moves toward G_t. Equivalent to TD(λ=1) with offline updates."
        ),
        intuition="Grade the whole exam at the end instead of after every question.",
        update_rule="G_t = r_{t+1} + γ G_{t+1};  Q(s_t,a_t) ← Q(s_t,a_t) + α [ G_t − Q(s_t,a_t) ]",
        pseudocode=(
            "for each episode:\n"
            "    play with eps-greedy(Q), storing (s, a, r)\n"
            "    G = 0\n"
            "    for (s, a, r) in reversed(episode):\n"
            "        G = r + gamma*G\n"
            "        Q[s,a] += alpha*(G - Q[s,a])"
        ),
        lifecycle="Act for a full episode, then sweep backwards over it once.",
        learns="A Q-table estimated from complete returns.",
        exploration="Linear epsilon-greedy decay per episode.",
        networks="None - lookup table.",
        failure_modes=[
            "Nothing is learned until an episode terminates",
            "High variance on long or stochastic episodes",
            "Useless on continuing tasks without a natural end",
        ],
        expected_metrics=_TABULAR_METRICS,
        recommended_envs=["blackjack", "gridworld", "frozen_lake"],
        related=["sarsa", "reinforce"],
        paper="Sutton & Barto, 2018. Reinforcement Learning: An Introduction, ch. 5.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        super().__init__(obs_dim, act_dim, params, device)
        self._episode = []

    def update(self, obs, action, reward, next_obs, done):
        self._episode.append((_state_key(obs), int(action), float(reward)))

    def end_of_episode(self):
        g = 0.0
        for s, a, r in reversed(self._episode):
            g = r + self.gamma * g
            td = g - self.Q[s][a]
            self.Q[s][a] += self.alpha * td
            self.td_errors.append(abs(td))
        self._episode = []
        self._anneal()


# --------------------------------------------------------------------------- #
_BANDIT_METRICS = [
    {"key": "table_size", "meaning": "Distinct states seen (1 for a pure bandit)."},
    {"key": "mean_bonus", "meaning": "Average exploration bonus of the chosen action."},
]


class UCBAgent(_TabularBase):
    SPEC = build_spec(
        name="ucb",
        label="UCB (upper confidence bound)",
        description="Exploration by optimism: prefer actions whose value is uncertain "
        "because they have rarely been tried. No epsilon to tune.",
        category="bandit", policy_type="off-policy",
        params=[
            ParamSpec("c", "Exploration constant", 2.0, "float",
                      "Width of the confidence bonus. Larger explores more.", min=0.0, max=20.0),
            ParamSpec("alpha", "Step size", 0.0, "float",
                      "0 uses exact sample averages (best for stationary problems); "
                      "a constant like 0.1 tracks drifting rewards.", min=0.0, max=1.0),
            ParamSpec("gamma", "Discount factor", 0.0, "float",
                      "0 for bandits. Above 0 bootstraps like Q-learning (UCB-Q).",
                      min=0.0, max=0.9999),
        ],
        presets=[
            {"name": "bandit", "label": "Bandit", "description": "UCB1 on a stationary bandit.",
             "overrides": {}, "episodes": 1000},
            {"name": "mdp", "label": "Small MDP", "description": "Optimistic Q-learning.",
             "overrides": {"gamma": 0.99, "alpha": 0.1, "c": 0.5}, "episodes": 400},
        ],
        beginner=(
            "Instead of exploring at random, UCB gives every action a bonus that "
            "shrinks the more often it is tried. Rarely-tried actions look attractive "
            "until the agent has enough evidence about them."
        ),
        technical=(
            "Chooses argmax_a Q(s,a) + c √(ln N(s) / N(s,a)), trying untried actions "
            "first (UCB1, Auer et al. 2002). Q uses sample averages when α=0. With "
            "γ>0 the target bootstraps from max_a′ Q(s′,a′), giving a simple "
            "count-based optimistic Q-learning for small MDPs."
        ),
        intuition="Optimism in the face of uncertainty: if you do not know, assume it "
        "might be great and go and find out.",
        update_rule="a = argmax_a [ Q(s,a) + c √( ln N(s) / N(s,a) ) ]",
        pseudocode=(
            "for each step:\n"
            "    if some action untried in s: a = that action\n"
            "    else: a = argmax Q[s] + c*sqrt(log(N[s]) / N[s,:])\n"
            "    r, s2 = env.step(a);  N[s,a] += 1\n"
            "    Q[s,a] += (1/N[s,a]) * (r + gamma*max(Q[s2]) - Q[s,a])"
        ),
        learns="Action values plus visit counts per state.",
        exploration="Deterministic confidence bonus; no randomness at all.",
        networks="None - lookup table.",
        failure_modes=[
            "Counts do not generalise: hopeless with large or continuous state spaces",
            "Sample averages adapt poorly when reward distributions drift (use α>0)",
        ],
        expected_metrics=_BANDIT_METRICS,
        recommended_envs=["multi_armed_bandit", "nonstationary_bandit", "river_swim"],
        related=["thompson", "q_learning"],
        paper="Auer, Cesa-Bianchi & Fischer, 2002. Finite-time analysis of the multiarmed bandit problem.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        p = dict(params)
        p.setdefault("epsilon_start", 0.0)
        p.setdefault("epsilon_end", 0.0)
        p.setdefault("epsilon_decay_episodes", 1)
        p.setdefault("alpha", 0.0)
        super().__init__(obs_dim, act_dim, p, device)
        self.c = float(p["c"])
        self.N: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.act_dim))
        self.bonuses = deque(maxlen=500)

    def get_action(self, obs, valid_actions=None):
        _, mask = self._unpack(obs)
        actions = self._mask_or_range(mask if mask is not None else valid_actions)
        s = _state_key(obs)
        n = self.N[s]
        untried = [a for a in actions if n[a] == 0]
        if untried and self.TRAIN:
            return int(untried[0])
        if not self.TRAIN:
            return _greedy(self.Q[s], actions)
        total = max(float(n[actions].sum()), 1.0)
        bonus = {a: self.c * np.sqrt(np.log(total) / max(n[a], 1)) for a in actions}
        best = max(actions, key=lambda a: self.Q[s][a] + bonus[a])
        self.bonuses.append(bonus[best])
        return int(best)

    def update(self, obs, action, reward, next_obs, done):
        s, s2, a = _state_key(obs), _state_key(next_obs), int(action)
        self.N[s][a] += 1
        step = self.alpha if self.alpha > 0 else 1.0 / self.N[s][a]
        target = float(reward)
        if not done and self.gamma > 0:
            target += self.gamma * float(np.max(self.Q[s2]))
        td = target - self.Q[s][a]
        self.Q[s][a] += step * td
        self.td_errors.append(abs(td))

    def get_metrics(self):
        return {
            "table_size": len(self.Q),
            "mean_bonus": round(float(np.mean(self.bonuses)), 4) if self.bonuses else 0.0,
            "td_error": round(float(np.mean(self.td_errors)), 5) if self.td_errors else 0.0,
        }


class ThompsonSamplingAgent(UCBAgent):
    SPEC = build_spec(
        name="thompson",
        label="Thompson sampling",
        description="Bayesian exploration: keep a belief about each action's value, "
        "sample from it, act greedily on the sample.",
        category="bandit", policy_type="off-policy",
        params=[
            ParamSpec("prior_std", "Prior std", 1.0, "float",
                      "Initial uncertainty about each action's mean reward.", min=0.01, max=100.0),
            ParamSpec("noise_std", "Reward noise std", 1.0, "float",
                      "Assumed standard deviation of observed rewards.", min=0.01, max=100.0),
        ],
        presets=[{"name": "bandit", "label": "Bandit", "description": "Gaussian Thompson sampling.",
                  "overrides": {}, "episodes": 1000}],
        beginner=(
            "The agent keeps a bell curve for each arm: its best guess of the payout "
            "and how unsure it is. Each turn it draws one random sample per arm and "
            "pulls whichever sample is highest. Uncertain arms sometimes draw high "
            "samples, so they still get tried, but less and less as evidence grows."
        ),
        technical=(
            "Gaussian-Gaussian conjugate model per (s,a): posterior precision τ = "
            "1/σ₀² + n/σ², posterior mean μ = (Σ r / σ²) / τ. Action = argmax of "
            "θ_a ~ N(μ_a, 1/τ_a). Probability matching (Thompson, 1933)."
        ),
        intuition="Act in proportion to the probability that each action is the best one.",
        update_rule="θ_a ~ N(μ_a, τ_a⁻¹);  a = argmax_a θ_a;  update (μ_a, τ_a) with r",
        pseudocode=(
            "for each step:\n"
            "    theta = sample N(mu[s], 1/tau[s]) for each action\n"
            "    a = argmax theta;  r = env.step(a)\n"
            "    tau[s,a] += 1/noise^2;  sum[s,a] += r/noise^2;  mu[s,a] = sum/tau"
        ),
        learns="A Gaussian posterior over the mean reward of every action.",
        exploration="Posterior sampling.",
        networks="None.",
        failure_modes=[
            "Assumes rewards are Gaussian and stationary",
            "Ignores future states: a bandit method, not a full MDP solver",
        ],
        expected_metrics=_BANDIT_METRICS,
        recommended_envs=["multi_armed_bandit", "probe5"],
        poor_envs=["gridworld", "cartpole"],
        related=["ucb"],
        paper="Thompson, 1933; Russo et al., 2018. A tutorial on Thompson sampling.",
    )

    def __init__(self, obs_dim, act_dim, params, device="cpu"):
        p = dict(params)
        p.setdefault("c", 0.0)
        p.setdefault("gamma", 0.0)
        super().__init__(obs_dim, act_dim, p, device)
        self.prior_prec = 1.0 / float(p["prior_std"]) ** 2
        self.noise_prec = 1.0 / float(p["noise_std"]) ** 2
        self.S: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.act_dim))

    def _posterior(self, s):
        prec = self.prior_prec + self.N[s] * self.noise_prec
        return self.S[s] * self.noise_prec / prec, prec

    def get_action(self, obs, valid_actions=None):
        _, mask = self._unpack(obs)
        actions = self._mask_or_range(mask if mask is not None else valid_actions)
        s = _state_key(obs)
        mu, prec = self._posterior(s)
        if not self.TRAIN:
            return _greedy(mu, actions)
        theta = np.random.normal(mu, 1.0 / np.sqrt(prec))
        self.bonuses.append(float(np.mean(1.0 / np.sqrt(prec[actions]))))
        return _greedy(theta, actions)

    def update(self, obs, action, reward, next_obs, done):
        s, a = _state_key(obs), int(action)
        self.N[s][a] += 1
        self.S[s][a] += float(reward)
        mu, _ = self._posterior(s)
        self.Q[s] = mu
        self.td_errors.append(abs(float(reward) - mu[a]))
