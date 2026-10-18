"""Adapters that normalize every supported environment type to a common API.

Observation handling
--------------------
Gymnasium exposes a wide variety of observation spaces. Everything is
flattened into a single float vector so the agents only ever see fixed-size
inputs:

=====================  ================================================
Space                  Encoding
=====================  ================================================
``Discrete(n)``        one-hot vector of length ``n``
``MultiBinary(n)``     the ``n`` bits as floats
``Box``                flattened values
``MultiDiscrete``      one-hot per component, concatenated
``Tuple``              each element encoded by its own space, concatenated
``Dict``               ``obs``/``observation`` encoded; ``action_mask``
                       is passed through for mask-aware agents
=====================  ================================================
"""

import numpy as np


def _box_size(space) -> int:
    shape = getattr(space, "shape", None)
    return int(np.prod(shape)) if shape else 1


def _space_size(space) -> int:
    """Number of floats needed to encode one point of `space`."""
    import gymnasium.spaces as spaces

    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.MultiBinary):
        return int(np.prod(space.shape)) if getattr(space, "shape", None) else int(space.n)
    if isinstance(space, spaces.MultiDiscrete):
        return int(sum(int(n) for n in space.nvec))
    if isinstance(space, spaces.Box):
        return _box_size(space)
    if isinstance(space, spaces.Tuple):
        return int(sum(_space_size(s) for s in space.spaces))
    if isinstance(space, spaces.Dict):
        return _space_size(_dict_obs_space(space))
    # Unknown space - fall back to a single scalar.
    shape = getattr(space, "shape", None)
    return int(np.prod(shape)) if shape else 1


def _dict_obs_space(space):
    """Pick the sub-space that actually carries the observation payload."""
    for key in ("obs", "observation", "state", "image"):
        if key in space.spaces:
            return space.spaces[key]
    return next(iter(space.spaces.values()))


def _encode(space, value) -> np.ndarray:
    """Encode one observation drawn from `space` as a flat float32 vector."""
    import gymnasium.spaces as spaces

    if isinstance(space, spaces.Discrete):
        vec = np.zeros(int(space.n), dtype=np.float32)
        idx = int(value)
        if 0 <= idx < int(space.n):
            vec[idx] = 1.0
        return vec

    if isinstance(space, spaces.MultiBinary):
        return np.asarray(value, dtype=np.float32).flatten()

    if isinstance(space, spaces.MultiDiscrete):
        parts = []
        for n, v in zip(space.nvec, np.asarray(value).flatten()):
            n = int(n)
            vec = np.zeros(n, dtype=np.float32)
            idx = int(v)
            if 0 <= idx < n:
                vec[idx] = 1.0
            parts.append(vec)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    if isinstance(space, spaces.Box):
        return np.asarray(value, dtype=np.float32).flatten()

    if isinstance(space, spaces.Tuple):
        flat_value = list(value)
        parts = [
            _encode(s, v) for s, v in zip(space.spaces, flat_value)
        ]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    if isinstance(space, spaces.Dict):
        payload = None
        for key in ("obs", "observation", "state", "image"):
            if key in value:
                payload = (space.spaces[key], value[key])
                break
        if payload is None:
            first_key = next(iter(space.spaces.keys()))
            payload = (space.spaces[first_key], value[first_key])
        return _encode(*payload)

    # Unknown space: best effort.
    return np.asarray(value, dtype=np.float32).flatten()


def _extract_mask(obs):
    """Return a legal-action mask if the observation carries one."""
    if isinstance(obs, dict):
        for key in ("action_mask", "legal_moves"):
            if obs.get(key) is not None:
                return [int(bool(v)) for v in np.asarray(obs[key]).flatten()]
    return None


class GymAdapter:
    """Gymnasium adapter. Flattens observations; one-hot encodes Discrete spaces."""

    def __init__(self, env_name: str):
        import gymnasium as gym
        import gymnasium.spaces as spaces

        self._env = gym.make(env_name)
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        self._obs_space = obs_space
        self.obs_dim = _space_size(obs_space)
        self._obs_mode = "onehot" if isinstance(obs_space, spaces.Discrete) else "flat"

        # handle both discrete and continuous action spaces
        if isinstance(act_space, spaces.Discrete):
            self.act_dim = int(act_space.n)
            self._act_mode = "discrete"
        elif isinstance(act_space, spaces.Box):
            self.act_dim = _box_size(act_space)
            self._act_mode = "continuous"
            self._act_shape = tuple(act_space.shape) if act_space.shape else (1,)
            self.act_low = np.asarray(act_space.low, dtype=np.float32).flatten()
            self.act_high = np.asarray(act_space.high, dtype=np.float32).flatten()
        else:
            self.act_dim = int(getattr(act_space, "n", 1))
            self._act_mode = "discrete"

    def _convert(self, obs):
        vec = _encode(self._obs_space, obs)
        if vec.size == 0:
            vec = np.zeros(self.obs_dim, dtype=np.float32)
        return vec.tolist()

    def reset(self):
        result = self._env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        mask = _extract_mask(obs)
        if mask is not None:
            self._last_mask = mask
            return {"obs": self._convert(obs), "action_mask": mask}
        return self._convert(obs)

    def step(self, action):
        # handle continuous vs discrete
        if getattr(self, "_act_mode", "discrete") == "continuous":
            # Always hand the environment a properly shaped array: several
            # envs (e.g. MountainCarContinuous) index action[0] and break on a
            # bare scalar.
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if arr.size != self.act_dim:
                arr = np.zeros(self.act_dim, dtype=np.float32)
            arr = np.clip(arr, self.act_low, self.act_high)
            act = arr.reshape(getattr(self, "_act_shape", (self.act_dim,)))
            obs, reward, terminated, truncated, info = self._env.step(act)
        else:
            obs, reward, terminated, truncated, info = self._env.step(int(action))

        mask = _extract_mask(obs)
        if mask is not None:
            self._last_mask = mask
            obs_out = {"obs": self._convert(obs), "action_mask": mask}
        else:
            obs_out = self._convert(obs)

        return (
            obs_out,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


class MinAtarAdapter:
    """MinAtar compact arcade games (state() returns a C x H x W binary array)."""

    def __init__(self, game: str):
        from minatar import Environment as MinAtarEnv

        self._env = MinAtarEnv(game)
        s = self._env.state()
        self.obs_dim = int(np.prod(s.shape))
        self.act_dim = int(
            self._env.num_actions()
            if callable(self._env.num_actions)
            else self._env.num_actions
        )
        self._shape = s.shape

    def _convert(self, state):
        return np.asarray(state, dtype=np.float32).flatten().tolist()

    def reset(self):
        self._env.reset()
        return self._convert(self._env.state())

    def step(self, action):
        reward, terminated = self._env.act(int(action))
        return (
            self._convert(self._env.state()),
            float(reward),
            bool(terminated),
            False,
            {},
        )

    def close(self):
        try:
            if hasattr(self._env, "close"):
                self._env.close()
        except Exception:
            pass


class PettingZooSelfPlayAdapter:
    """A two-player PettingZoo classic game seen from one seat.

    The learner controls ``learner_seat``; every other seat is played by a
    scripted opponent that picks uniformly among legal moves. Each ``step()``
    is one learner move followed by opponent moves until it is the learner's
    turn again (or the game ends). Reward is the outcome *from the learner's
    seat*: +1 win, 0 draw, -1 loss (chip deltas for poker).

    The earlier version summed rewards across both seats of a zero-sum game,
    which made the training signal identically zero. Kept under the old class
    name so saved references and imports keep working.
    """

    _GAMES = {
        "tictactoe": ("pettingzoo.classic", "tictactoe_v3"),
        "connect_four": ("pettingzoo.classic", "connect_four_v3"),
        "texas_holdem": ("pettingzoo.classic", "texas_holdem_v4"),
    }

    def __init__(self, game: str, learner_seat: int = 0, seed: int | None = None):
        import importlib

        if game not in self._GAMES:
            raise ValueError(f"Unsupported pettingzoo game: {game}")
        package, module = self._GAMES[game]
        mod = importlib.import_module(f"{package}.{module}")
        self._env = mod.env()
        self._rng = np.random.default_rng(seed)
        self._env.reset(seed=seed)
        self._seat = self._env.possible_agents[learner_seat]
        act_space = self._env.action_space(self._seat)
        sample_obs = self._extract(self._env.observe(self._seat))[0]
        self.obs_dim = len(sample_obs)
        self.act_dim = int(act_space.n)
        self.last_outcome = None

    @staticmethod
    def _extract(obs):
        """Return (flat_obs_list, action_mask list or None) from any pettingzoo obs."""
        mask = None
        if isinstance(obs, dict):
            for key in ("action_mask", "legal_moves"):
                if key in obs and obs[key] is not None:
                    mask = obs[key]
                    break
            raw = obs.get("obs", obs.get("observation"))
            if raw is None:
                raw = next(iter(obs.values()))
        else:
            raw = obs
        flat = np.asarray(raw, dtype=np.float32).flatten().tolist()
        if mask is not None:
            mask = [int(bool(v)) for v in np.asarray(mask).flatten()]
        return flat, mask

    def _done(self) -> bool:
        return not self._env.agents or all(
            self._env.terminations.get(a, False) or self._env.truncations.get(a, False)
            for a in self._env.agents
        )

    def _learner_reward(self) -> float:
        """Terminal reward for the learner's seat (all classic games pay at the end)."""
        rewards = getattr(self._env, "rewards", {}) or {}
        cum = getattr(self._env, "_cumulative_rewards", {}) or {}
        return float(rewards.get(self._seat, 0.0) or cum.get(self._seat, 0.0) or 0.0)

    def _advance_opponents(self):
        """Play scripted moves until it is the learner's turn or the game ends."""
        while not self._done() and self._env.agent_selection != self._seat:
            agent = self._env.agent_selection
            if self._env.terminations.get(agent) or self._env.truncations.get(agent):
                self._env.step(None)
                continue
            _, mask = self._extract(self._env.observe(agent))
            legal = [i for i, v in enumerate(mask or []) if v] or list(range(self.act_dim))
            self._env.step(int(self._rng.choice(legal)))

    def _observation(self):
        flat, mask = self._extract(self._env.observe(self._seat))
        return {"obs": flat, "action_mask": mask or [1] * self.act_dim}

    def reset(self):
        self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        self._episode_reward = 0.0
        self.last_outcome = None
        self._advance_opponents()
        return self._observation()

    def step(self, action):
        if not self._done():
            self._env.step(int(action))
            self._advance_opponents()
        done = self._done()
        reward = self._learner_reward() if done else 0.0
        if done:
            self.last_outcome = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
        return self._observation_or_blank(done), reward, done, False, {"outcome": self.last_outcome}

    def _observation_or_blank(self, done: bool):
        if done:
            return {"obs": [0.0] * self.obs_dim, "action_mask": [1] * self.act_dim}
        return self._observation()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


class BuiltinAdapter:
    """Wraps builtin pure-python envs (already close to the target API)."""

    def __init__(self, factory):
        self._env = factory()

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def obs_dim(self):
        return self._env.obs_dim

    @property
    def act_dim(self):
        return self._env.act_dim

    def step(self, action):
        # Continuous builtin envs (e.g. probe6) take a float vector; the rest
        # take an integer action index.
        if getattr(self._env, "continuous", False):
            action = np.clip(np.asarray(action, dtype=np.float32).flatten(),
                             self._env.act_low, self._env.act_high)
        else:
            action = int(action)
        out = self._env.step(action)
        if isinstance(out, dict):  # probe-style single-return
            return out, 0.0, True, False, {}
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            if not isinstance(obs, dict):
                obs = {
                    "obs": list(np.atleast_1d(obs).astype(float)),
                    "action_mask": [1] * self._env.act_dim,
                }
            return obs, float(reward), bool(terminated), bool(truncated), info
        raise ValueError(f"Unexpected step output: {type(out)}")

    def reset(self):
        obs = self._env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if not isinstance(obs, dict):
            obs = {
                "obs": list(np.atleast_1d(obs).astype(float)),
                "action_mask": [1] * self._env.act_dim,
            }
        return obs

    def close(self):
        pass
