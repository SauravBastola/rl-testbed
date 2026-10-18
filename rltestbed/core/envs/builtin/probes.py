"""Diagnostic probe environments (pure python, gym-like API).

These are tiny environments designed to verify that an RL implementation
can learn specific things. Classic sanity checks before scaling up.
"""

import random


class _ProbeBase:
    """Minimal gym-like base for probes with dict obs + action mask."""

    obs_dim = 1
    act_dim = 2

    def reset(self):
        self._state = 0
        return {"obs": [0], "action_mask": [1] * self.act_dim}

    def close(self):
        pass

    def _out(self, reward, done):
        return (
            {"obs": [self._state], "action_mask": [1] * self.act_dim},
            reward,
            done,
            False,
            {},
        )


class ProbeEnv1(_ProbeBase):
    """Single state, 1 action, constant reward 1. Trivial policy check."""

    obs_dim, act_dim = 1, 1

    def step(self, action):
        self._state = 1
        return self._out(1.0, True)


class ProbeEnv2(_ProbeBase):
    """State-dependent reward: action 1 is always better. Tests conditioning."""

    obs_dim, act_dim = 1, 2

    def step(self, action):
        self._state = 1
        return self._out(1.0 if action == 1 else 0.0, True)


class ProbeEnv3(_ProbeBase):
    """Delayed reward: sequence [0, 1] yields +1 on the second step.
    Tests credit assignment and discounting (gamma)."""

    obs_dim, act_dim = 1, 2

    def reset(self):
        self._state = 0
        self._steps = 0
        return {"obs": [self._state], "action_mask": [1, 1]}

    def step(self, action):
        self._steps += 1
        if self._state == 0:
            if action == 0:
                self._state = 1
                return self._out(0.0, False)
            self._state = -1
            return self._out(0.0, True)
        # state == 1: need action 1
        reward = 1.0 if action == 1 else 0.0
        return self._out(reward, True)


class ProbeEnv4(_ProbeBase):
    """Stochastic bandit in a single state; action 1 pays off with p=0.8."""

    def step(self, action):
        self._state = 1
        p = 0.8 if action == 1 else 0.2
        return self._out(1.0 if random.random() < p else 0.0, True)


class ProbeEnv5(_ProbeBase):
    """Two-armed bandit: arm means are 0.2 and 0.8. Exploration/exploitation test."""

    obs_dim = 1

    def step(self, action):
        p = 0.8 if action == 1 else 0.2
        return self._out(1.0 if random.random() < p else 0.0, True)


PROBES = {
    "probe1": (ProbeEnv1, "One-state constant reward"),
    "probe2": (ProbeEnv2, "State-conditioned reward"),
    "probe3": (ProbeEnv3, "Delayed reward chain"),
    "probe4": (ProbeEnv4, "Stochastic transitions"),
    "probe5": (ProbeEnv5, "Two-armed bandit"),
}
