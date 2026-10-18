"""Example custom environment: SignBandit.

A one-armed bandit whose best action flips every episode. Copy this file,
rename things, and start experimenting.
"""

import random

ENV_SPEC = {
    "name": "sign_bandit",
    "label": "Sign Bandit",
    "description": "Two-armed bandit where the better arm is chosen randomly each episode. "
    "Tests fast adaptation within episodes.",
    "max_steps": 50,
    "tags": ["bandit", "example"],
    "difficulty": "easy",
}


class SignBandit:
    obs_dim = 2  # [episode_progress, current_best_arm_hint]
    act_dim = 2

    def reset(self):
        self._good = random.randrange(2)
        self._steps = 0
        return {"obs": [0.0, float(self._good)], "action_mask": [1, 1]}

    def step(self, action):
        self._steps += 1
        reward = 1.0 if action == self._good else -0.25
        done = self._steps >= ENV_SPEC["max_steps"]
        return (
            {
                "obs": [self._steps / ENV_SPEC["max_steps"], float(self._good)],
                "action_mask": [1, 1],
            },
            reward,
            done,
            False,
            {},
        )

    def close(self):
        pass


def make():
    return SignBandit()
