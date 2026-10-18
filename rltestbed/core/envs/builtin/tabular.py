"""Builtin tabular / exploration benchmark environments (pure python).

These four environments are dependency-free and deliberately target the
*conceptual* difficulties in RL rather than raw simulation fidelity:

``MultiArmedBanditEnv``
    The exploration-exploitation trade-off with no state at all.

``RiverSwimEnv``
    Hard-exploration benchmark: greedy policies get trapped collecting a tiny
    reward near the start and never discover the large reward upstream.

``NChainEnv``
    Long-horizon credit assignment: the myopic action pays immediately, the
    optimal action pays much more but only after a chain of decisions.

``WindyGridworldEnv``
    Sutton & Barto's classic gridworld with a column-dependent crosswind,
    used to demonstrate that the agent must learn to compensate for dynamics.

All of them expose the testbed's builtin interface:
``obs_dim``, ``act_dim``, ``reset()``, ``step(action)``, ``close()``,
and a text ``render()`` used by the web preview.
"""

import random


class _TabularBase:
    """Shared plumbing for the one-hot encoded tabular environments."""

    def _onehot(self, index: int, size: int):
        vec = [0.0] * size
        vec[int(index)] = 1.0
        return vec

    def _out(self, obs, reward, terminated, truncated=False, mask=None):
        if mask is None:
            mask = [1] * self.act_dim
        return {"obs": obs, "action_mask": mask}, float(reward), bool(terminated), bool(truncated), {}

    def close(self):
        pass


class MultiArmedBanditEnv(_TabularBase):
    """k-armed bandit. One state, k actions, Gaussian rewards.

    The cleanest possible statement of the exploration-exploitation dilemma:
    there is no state to generalise over, so every gain must come from
    choosing what to try next.

    Parameters
    ----------
    k : number of arms
    nonstationary : if True the true action values perform a small random walk
        after every pull, so the agent must keep exploring forever.
    horizon : pulls per episode (default 1 - one pull, one episode).
    reward_std : standard deviation of the reward noise.
    """

    def __init__(self, k: int = 10, nonstationary: bool = False,
                 horizon: int = 1, reward_std: float = 1.0, seed=None):
        self.k = k
        self.nonstationary = nonstationary
        self.horizon = horizon
        self.reward_std = reward_std
        self.act_dim = k
        self.obs_dim = 1
        self._rng = random.Random(seed)
        self._q = [self._rng.gauss(0.0, 1.0) for _ in range(k)]
        self._best = max(range(k), key=lambda i: self._q[i])
        self._t = 0
        self.reset()

    def reset(self):
        self._t = 0
        # A single constant "state" - there is genuinely nothing to observe.
        return self._out([1.0], 0.0, False)

    def step(self, action):
        a = int(action)
        reward = self._rng.gauss(self._q[a], self.reward_std)
        if self.nonstationary:
            # True values drift, so the optimal arm changes over time.
            self._q = [q + self._rng.gauss(0.0, 0.01) for q in self._q]
        self._t += 1
        terminated = self._t >= self.horizon
        return self._out([1.0], reward, terminated, False)

    def render(self, obs=None) -> str:
        bars = []
        for i, q in enumerate(self._q):
            bars.append(f"  arm {i:2d}  q={q:+.3f}  {'#' * max(1, int(abs(q) * 8))}")
        best = self._best
        return (
            f"{self.k}-armed bandit"
            f"{' (non-stationary)' if self.nonstationary else ''}\n"
            + "\n".join(bars)
            + f"\n  best arm: {best}   pulls this episode: {self._t}/{self.horizon}"
        )


class RiverSwimEnv(_TabularBase):
    """RiverSwim - the canonical hard-exploration benchmark.

    A swimmer starts at the left bank of a river and can swim left or right.
    Swimming left from the start state yields a tiny guaranteed reward
    (``r_small``); swimming right is uncertain, but leads upstream to a state
    where a much larger reward (``1.0``) can be collected repeatedly.

    The trap is intentional: an epsilon-greedy agent almost always settles on
    the small reward and never discovers the big one. It is the standard
    environment for demonstrating *deep* / optimism-driven exploration.
    """

    def __init__(self, n_states: int = 6, r_small: float = 0.005,
                 p_right: float = 0.6, max_steps: int = 100, seed=None):
        self.n_states = n_states
        self.r_small = r_small
        self.p_right = p_right
        self.max_steps = max_steps
        self.act_dim = 2          # 0 = swim left, 1 = swim right
        self.obs_dim = n_states
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.state = 0
        self._t = 0
        return self._out(self._onehot(self.state, self.n_states), 0.0, False)

    def step(self, action):
        a = int(action)
        reward = 0.0
        if a == 0:  # swim left
            if self.state == 0:
                reward = self.r_small          # guaranteed trickle at the bank
            else:
                self.state -= 1
        else:       # swim right
            if self.state == self.n_states - 1:
                # At the far bank: usually stay and collect, sometimes slip back.
                if self._rng.random() < self.p_right:
                    pass
                else:
                    self.state -= 1
                reward = 1.0
            else:
                if self._rng.random() < self.p_right:
                    self.state += 1
        self._t += 1
        truncated = self._t >= self.max_steps
        return self._out(self._onehot(self.state, self.n_states), reward, False, truncated)

    def render(self, obs=None) -> str:
        cells = []
        for s in range(self.n_states):
            if s == self.state:
                cells.append("<A>")
            elif s == self.n_states - 1:
                cells.append(" $ ")
            else:
                cells.append(" ~ ")
        return (
            "RiverSwim  (left bank  ->  upstream reward)\n"
            + "".join(cells)
            + f"\nstate={self.state}/{self.n_states - 1}   step={self._t}/{self.max_steps}"
            + f"\nactions: 0 = swim left (safe {self.r_small}), 1 = swim right (reach "
            f"the far bank for +1.0)"
        )


class NChainEnv(_TabularBase):
    """N-chain: long-horizon credit assignment.

    The agent moves along a chain of ``n`` states. At every step it can either
    advance (action 0) or jump back to the start (action 1). Jumping back pays
    a small immediate reward (``r_small``); advancing pays nothing until the
    end of the chain, where each advance pays ``r_big``.

    With ``slip > 0``, advancing sometimes fails, which makes the chain even
    longer in expectation. The optimal policy is to advance forever, but a
    myopic agent will keep taking the small reward - the classic demonstration
    of why discounting and exploration both matter.
    """

    def __init__(self, n: int = 5, slip: float = 0.2, r_small: float = 2.0,
                 r_big: float = 10.0, max_steps: int = 100, seed=None):
        self.n = n
        self.slip = slip
        self.r_small = r_small
        self.r_big = r_big
        self.max_steps = max_steps
        self.act_dim = 2          # 0 = advance, 1 = jump back to start
        self.obs_dim = n
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.state = 0
        self._t = 0
        return self._out(self._onehot(self.state, self.n), 0.0, False)

    def step(self, action):
        a = int(action)
        if a == 1:  # bail out: small immediate reward, back to the start
            self.state = 0
            reward = self.r_small
        else:       # advance along the chain
            if self.state == self.n - 1:
                reward = self.r_big        # collect at the end of the chain
            else:
                reward = 0.0
                if self._rng.random() >= self.slip:
                    self.state += 1
        self._t += 1
        truncated = self._t >= self.max_steps
        return self._out(self._onehot(self.state, self.n), reward, False, truncated)

    def render(self, obs=None) -> str:
        cells = []
        for s in range(self.n):
            if s == self.state:
                cells.append("[A]")
            elif s == self.n - 1:
                cells.append("[*]")
            else:
                cells.append("[ ]")
        return (
            "N-Chain\n" + "-" + "-".join(cells) + "->\n"
            + f"state={self.state}/{self.n - 1}   step={self._t}/{self.max_steps}\n"
            + f"actions: 0 = advance (reach the end for +{self.r_big:g}), "
            f"1 = jump back to start (+{self.r_small:g})"
        )


class WindyGridworldEnv(_TabularBase):
    """Windy gridworld (Sutton & Barto, Example 6.5).

    A 7x10 grid with an updraft that varies by column. Every move is displaced
    upward by the wind strength of the column, so the agent must learn to
    aim *into* the wind to make progress. Undiscounted, reward -1 per step.

    The point is not exploration but control: the optimal path deliberately
    overshoots and lets the wind carry the agent to the goal.
    """

    # Updraft strength for each of the 10 columns.
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    START = (3, 0)
    GOAL = (3, 7)

    def __init__(self, rows: int = 7, cols: int = 10, max_steps: int = 200,
                 stochastic: bool = False, seed=None):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.act_dim = 4          # 0 = up, 1 = down, 2 = left, 3 = right
        self.obs_dim = rows * cols
        self._rng = random.Random(seed)
        self.reset()

    def _wind(self, col: int) -> int:
        w = self.WIND[col % len(self.WIND)]
        if self.stochastic:
            w += self._rng.choice((-1, 0, 1))
        return max(0, w)

    def reset(self):
        self.pos = self.START
        self._t = 0
        return self._out(self._onehot(self._idx(self.pos), self.obs_dim), 0.0, False)

    def _idx(self, pos):
        return pos[0] * self.cols + pos[1]

    def step(self, action):
        r, c = self.pos
        a = int(action)
        if a == 0:
            r -= 1
        elif a == 1:
            r += 1
        elif a == 2:
            c -= 1
        elif a == 3:
            c += 1
        # The crosswind pushes the agent upward, on top of its own move.
        r -= self._wind(c)
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))
        self.pos = (r, c)
        self._t += 1
        terminated = self.pos == self.GOAL
        truncated = self._t >= self.max_steps and not terminated
        reward = 0.0 if terminated else -1.0
        return self._out(self._onehot(self._idx(self.pos), self.obs_dim),
                         reward, terminated, truncated)

    def render(self, obs=None) -> str:
        rows = []
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                if (r, c) == self.pos:
                    cells.append("A")
                elif (r, c) == self.GOAL:
                    cells.append("G")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return (
            "Windy Gridworld\n"
            + "\n".join(rows)
            + "\nwind  " + " ".join(str(w) for w in self.WIND[: self.cols])
            + f"\npos={self.pos}  step={self._t}/{self.max_steps}"
        )


class DynaMazeEnv(_TabularBase):
    """Sutton & Barto's Dyna maze (Figure 8.2): a 6x9 room with three walls.

    Reward is 0 everywhere except +1 on reaching the goal, so a model-free
    learner propagates value backwards one step per episode while Dyna-Q's
    planning spreads it through the whole maze between real steps.
    """

    ROWS, COLS = 6, 9
    START, GOAL = (2, 0), (0, 8)
    WALLS = {(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def __init__(self, max_steps: int = 1000):
        self.obs_dim = self.ROWS * self.COLS
        self.act_dim = 4
        self.max_steps = max_steps
        self.reset()

    def _idx(self, pos):
        return pos[0] * self.COLS + pos[1]

    def reset(self):
        self.pos, self.t = self.START, 0
        return self._out(self._onehot(self._idx(self.pos), self.obs_dim), 0.0, False)[0]

    def step(self, action):
        dr, dc = self.MOVES[int(action)]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.ROWS and 0 <= c < self.COLS and (r, c) not in self.WALLS:
            self.pos = (r, c)
        self.t += 1
        done = self.pos == self.GOAL
        return self._out(self._onehot(self._idx(self.pos), self.obs_dim),
                         1.0 if done else 0.0, done, self.t >= self.max_steps and not done)

    def render(self, obs=None) -> str:
        rows = []
        for r in range(self.ROWS):
            line = ""
            for c in range(self.COLS):
                p = (r, c)
                line += ("A" if p == self.pos else "G" if p == self.GOAL
                         else "#" if p in self.WALLS else ".")
            rows.append(line)
        return "\n".join(rows)


class NonstationaryBanditEnv(MultiArmedBanditEnv):
    """10-armed bandit whose true arm values random-walk after every pull."""

    def __init__(self, **kwargs):
        kwargs.setdefault("nonstationary", True)
        super().__init__(**kwargs)


TABULAR_ENVS = {
    "dyna_maze": DynaMazeEnv,
    "nonstationary_bandit": NonstationaryBanditEnv,
    "multi_armed_bandit": MultiArmedBanditEnv,
    "river_swim": RiverSwimEnv,
    "n_chain": NChainEnv,
    "windy_gridworld": WindyGridworldEnv,
}

__all__ = [
    "MultiArmedBanditEnv",
    "RiverSwimEnv",
    "NChainEnv",
    "WindyGridworldEnv",
    "TABULAR_ENVS",
]
