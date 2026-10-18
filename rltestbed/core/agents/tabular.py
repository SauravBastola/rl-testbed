"""Tabular control methods: Q-learning and SARSA.

These maintain an explicit state-action value table. They are the clearest way
to *see* RL working: with one-hot observations the table index is just the
argmax of the observation vector. Best paired with Gridworld / Catch-style
small discrete environments.
"""

from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np

from .base import Agent, ParamSpec, build_spec

_COMMON_PARAMS = [
    ParamSpec("alpha", "Learning rate", 0.1, "float",
              "Step size for table updates - how fast values move toward targets",
              min=0.001, max=1.0),
    ParamSpec("gamma", "Discount factor", 0.99, "float", "Reward discount", min=0.0, max=0.9999),
    ParamSpec("epsilon_start", "Epsilon start", 1.0, "float", "Initial exploration rate"),
    ParamSpec("epsilon_end", "Epsilon end", 0.05, "float", "Final exploration rate"),
    ParamSpec("epsilon_decay_episodes", "Epsilon decay episodes", 200, "int",
              "Episodes over which epsilon decays linearly", min=1),
]


def _state_key(obs) -> int:
    """Map a one-hot / integer observation to a table index."""
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)
    arr = np.asarray(obs, dtype=np.float32).flatten()
    amax = int(np.argmax(arr))
    if abs(float(arr[amax]) - 1.0) < 1e-6 and np.count_nonzero(arr) == 1:
        return amax
    # fallback: hash continuous vectors into a coarse grid
    return hash(tuple((arr * 10).round().astype(int).tolist())) & 0xFFFFF


def _spec(name, label, description, on_policy: bool):
    algo = "SARSA" if on_policy else "Q-learning"
    return build_spec(
        name=name,
        label=label,
        description=description,
        category="value-based",
        policy_type="on-policy" if on_policy else "off-policy",
        status="implemented",
        params=_COMMON_PARAMS,
        presets=[
            {"name": "beginner-default", "label": "Beginner default",
             "description": "Classic settings that solve Gridworld quickly.",
             "overrides": {}, "episodes": 300},
            {"name": "quick-demo", "label": "Quick demo",
             "description": "Fast convergence on tiny environments.",
             "overrides": {"alpha": 0.2}, "episodes": 150},
            {"name": "fast-iteration", "label": "Fast iteration",
             "description": "Short runs for exploring exploration parameters.",
             "overrides": {}, "episodes": 100},
        ],
        beginner=(
            "Instead of a neural network this agent keeps a literal table of numbers: "
            "one row per state, one column per action. After every step it nudges the "
            "entry for what it just did toward the reward it observed plus the value of "
            "the next state. "
            + ("SARSA learns from the action it *actually* takes next (including "
               "exploratory ones), making it more cautious."
               if on_policy else
               "Q-learning always learns toward the best imagined next action, whether "
               "or not it actually takes it - it is 'off-policy'.")
        ),
        technical=(
            "Update rule "
            + ("Q(s,a) ← Q(s,a) + α[r + γ·Q(s′,a′) − Q(s,a)] where a′ is the action "
               "actually selected next (on-policy TD control)."
               if on_policy else
               "Q(s,a) ← Q(s,a) + α[r + γ·max_a′ Q(s′,a′) − Q(s,a)] (off-policy TD control, "
               "Watkins' variant with Q(λ)=0 behaviour).")
            + " States are identified by the argmax of one-hot observations; unseen entries "
            "initialise to optimistic 0. Epsilon decays linearly per episode."
        ),
        intuition=(
            "The table is a cheat-sheet being graded after every single move. Values flow "
            "backwards from rewards one step at a time until early actions 'know' about "
            "distant goals."
        ),
        update_rule=(
            r"Q(s,a) ← Q(s,a) + α [ r + γ Q(s′,a′) − Q(s,a) ]" if on_policy else
            r"Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s′,a') − Q(s,a) ]"
        ),
        pseudocode=(
            "for each episode:\n"
            "    s = env.reset()\n"
            + ("    a = eps-greedy(Q[s])\n"
               "    while not done:\n"
               "        s2, r, done = env.step(a)\n"
               "        a2 = eps-greedy(Q[s2])\n"
               "        Q[s,a] += alpha*(r + gamma*Q[s2,a2] - Q[s,a])\n"
               "        s, a = s2, a2"
               if on_policy else
               "    while not done:\n"
               "        a = eps-greedy(Q[s])\n"
               "        s2, r, done = env.step(a)\n"
               "        Q[s,a] += alpha*(r + gamma*max(Q[s2]) - Q[s,a])\n"
               "        s = s2")
        ),
        lifecycle=(
            "1. Pick an action epsilon-greedily from the table row.\n"
            "2. Step the environment.\n"
            "3. Apply the one-step tabular update immediately.\n"
            "4. Anneal epsilon at the end of each episode."
        ),
        learns="An explicit |states| × |actions| table of Q-values.",
        exploration="Linear epsilon-greedy decay per episode.",
        replay_buffer="Not used - tabular updates are applied online.",
        target_network="Not used.",
        networks="None - pure lookup table (no torch parameters).",
        failure_modes=[
            "State space must be small; continuous inputs get coarse-hashed",
            f"{'On-policy: learns cautious paths that include exploration noise' if on_policy else 'Max operator can propagate overly-optimistic values in noisy environments'}",
            "Never explores states its policy avoids once epsilon is low",
        ],
        expected_metrics=[
            {"key": "epsilon", "meaning": "Exploration rate - decays per episode."},
            {"key": "table_size", "meaning": "Number of distinct states visited so far."},
            {"key": "td_error", "meaning": "Recent mean absolute TD error."},
        ],
        related=["q_learning", "sarsa", "dqn"],
    )


class _TabularBase(Agent):
    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.obs_dim = int(obs_dim)
        p = self.params
        self.Q: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.act_dim))
        self.alpha = float(p["alpha"])
        self.gamma = float(p["gamma"])
        self.epsilon = float(p["epsilon_start"])
        eps_end = max(float(p["epsilon_end"]), 0.0)
        slope = (float(p["epsilon_start"]) - eps_end) / max(int(p["epsilon_decay_episodes"]), 1)
        self._eps_end = eps_end
        self._eps_slope = slope
        self._episode_count = 0
        self.td_errors = deque(maxlen=200)

    def get_action(self, obs, valid_actions=None):
        _, mask = self._unpack(obs)
        actions = self._mask_or_range(mask if mask is not None else valid_actions)
        if len(actions) == 1:
            return actions[0]
        # Greedy evaluation (TRAIN=False) must not explore; this used to be
        # ignored, which made tabular eval scores noisy.
        if self.TRAIN and np.random.random() < self.epsilon:
            return int(np.random.choice(actions))
        q = self.Q[_state_key(obs)]
        best = max(actions, key=lambda a: q[a])
        return int(best)

    def _anneal(self):
        self._episode_count += 1
        self.epsilon = max(self._eps_end,
                           float(self.params["epsilon_start"]) - self._episode_count * self._eps_slope)

    def end_of_episode(self):
        self._anneal()

    def get_metrics(self):
        return {
            "epsilon": round(float(self.epsilon), 4),
            "table_size": len(self.Q),
            "td_error": round(float(np.mean(self.td_errors)), 5) if self.td_errors else 0.0,
        }

    def save_model(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "format": type(self).__name__,
                    "agent_name": self.SPEC["name"],
                    "Q": dict(self.Q),
                    "params": self.params,
                    "act_dim": self.act_dim,
                    "episode_count": self._episode_count,
                },
                f,
            )

    def load_model(self, path):
        import pickle
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.act_dim), ckpt.get("Q", {}))
        self._episode_count = ckpt.get("episode_count", 0)


class QLearningAgent(_TabularBase):
    SPEC = _spec("q_learning", "Tabular Q-learning",
                 "Off-policy tabular TD control - the clearest way to watch RL work. "
                 "Best on Gridworld and other small discrete tasks.", on_policy=False)

    def update(self, obs, action, reward, next_obs, done):
        s, s2 = _state_key(obs), _state_key(next_obs)
        _, mask = self._unpack(next_obs)
        valid = self._mask_or_range(mask)
        target = float(reward)
        if not done:
            target += self.gamma * max(self.Q[s2][a] for a in valid)
        td = target - self.Q[s][int(action)]
        self.Q[s][int(action)] += self.alpha * td
        self.td_errors.append(abs(td))


class SarsaAgent(_TabularBase):
    SPEC = _spec("sarsa", "SARSA",
                 "On-policy tabular TD control: learns the value of the policy it "
                 "actually follows, exploration included.", on_policy=True)

    def update(self, obs, action, reward, next_obs, done):
        s, s2 = _state_key(obs), _state_key(next_obs)
        _, mask = self._unpack(next_obs)
        valid = self._mask_or_range(mask)
        if done:
            target = float(reward)
        else:
            # choose the NEXT action the same way get_action will (incl. exploration)
            if np.random.random() < self.epsilon:
                a2 = int(np.random.choice(valid))
            else:
                a2 = int(max(valid, key=lambda a: self.Q[s2][a]))
            target = float(reward) + self.gamma * self.Q[s2][a2]
        td = target - self.Q[s][int(action)]
        self.Q[s][int(action)] += self.alpha * td
        self.td_errors.append(abs(td))
