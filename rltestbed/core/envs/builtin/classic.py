"""Builtin pure-python environments: gridworld navigation and catch game.

No pygame/rendering dependencies - designed for fast headless training.
"""

import random


class GridworldEnv:
    """NxN gridworld. Start top-left, reach the goal bottom-right.
    Step penalty -0.01, goal reward +1. Observation is one-hot-free index [row*N+col].
    """

    def __init__(self, size: int = 6, max_steps: int = 200):
        self.size = size
        self.max_steps = max_steps
        self.act_dim = 4
        self.obs_dim = size * size
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self._steps = 0
        return {"obs": self._onehot(), "action_mask": [1, 1, 1, 1]}

    def _onehot(self):
        vec = [0.0] * (self.size * self.size)
        vec[self.pos[0] * self.size + self.pos[1]] = 1.0
        return vec

    def step(self, action):
        r, c = self.pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.size - 1, c + 1)
        self.pos = (r, c)
        self._steps += 1
        terminated = self.pos == self.goal
        truncated = self._steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.01
        return (
            {"obs": self._onehot(), "action_mask": [1, 1, 1, 1]},
            reward,
            terminated,
            truncated,
            {},
        )

    def close(self):
        pass

    def render(self, obs=None) -> str:
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.pos:
                    row.append("A")
                elif (r, c) == self.goal:
                    row.append("★")
                else:
                    row.append("·")
            rows.append(" ".join(row))
        return "\n".join(rows)


class CatchGameEnv:
    """Catch: an item falls from a random column; move the paddle to catch it.
    Rewards: +1 catch, -1 miss. Observation: normalized [paddle_x, item_x, item_y]."""

    def __init__(self, columns: int = 10, max_steps: int = 100):
        self.columns = columns
        self.max_steps = max_steps
        self.act_dim = 3  # stay / left / right
        self.obs_dim = 3

    def reset(self):
        self.paddle = self.columns // 2
        self.item_col = random.randrange(self.columns)
        self.item_row = 0
        self._steps = 0
        return self._obs()

    def _obs(self):
        depth = float(self.columns)
        return [
            self.paddle / depth,
            self.item_col / depth,
            self.item_row / max(depth, 1),
        ]

    def step(self, action):
        if action == 1:
            self.paddle = max(0, self.paddle - 1)
        elif action == 2:
            self.paddle = min(self.columns - 1, self.paddle + 1)
        self.item_row += 1
        self._steps += 1

        if self.item_row >= self.columns:
            caught = abs(self.paddle - self.item_col) <= 1
            return self._obs(), (1.0 if caught else -1.0), True, False, {}

        truncated = self._steps >= self.max_steps
        return self._obs(), 0.0, False, truncated, {}

    def close(self):
        pass

    def render(self, obs=None) -> str:
        grid = [["·"] * self.columns for _ in range(self.columns)]
        if self.item_row < self.columns:
            grid[self.item_row][self.item_col] = "●"
        paddle_row = ["·"] * self.columns
        for c in range(max(0, self.paddle - 1), min(self.columns, self.paddle + 2)):
            paddle_row[c] = "═"
        paddle_row[self.paddle] = "▲"
        return "\n".join(" ".join(r) for r in grid + [paddle_row])
