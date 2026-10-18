"""Uniformly random agent - a baseline that needs no training."""

from typing import Any, Dict

import numpy as np

from .base import Agent, build_spec

SPEC = build_spec(
    name="random",
    label="Random",
    description="Uniformly random action selection. The environment sanity check and "
    "the score every learning algorithm must beat.",
    category="baseline",
    policy_type="baseline",
    status="implemented",
    supports={"obs_types": ["vector", "onehot", "image"], "action_types": ["discrete", "continuous"], "multi_agent": False, "action_mask": True, "recurrent": False},
    beginner=(
        "This agent flips coins. It never learns - and that is exactly why it is useful: "
        "it tells you what a 'no skill' score looks like on any environment, and it "
        "exercises every environment pathway without touching any learning code."
    ),
    technical="Samples uniformly from valid actions each step. No parameters, no state.",
    intuition="Every trained agent should comfortably beat this line; if it doesn't, "
    "something is broken.",
    pseudocode="for each step: action ~ Uniform(valid actions)",
    exploration="Maximum possible - pure randomness forever.",
    failure_modes=["None - but never mistake its score for a learned policy."],
    expected_metrics=[{"key": "episode_return", "meaning": "Return accumulated this episode."}],
)


class RandomAgent(Agent):
    SPEC = SPEC

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.obs_dim = obs_dim
        self._episode_reward = 0.0

    def get_action(self, obs, valid_actions=None):
        # Continuous envs announce their bounds through set_action_bounds(); the
        # previous version guessed from the observation length instead.
        low = getattr(self, "act_low", None)
        if low is not None:
            return np.random.uniform(low, self.act_high).astype(np.float32)
        _, mask = self._unpack(obs)
        if mask is None and valid_actions is not None:
            mask = [1 if i in set(valid_actions) else 0 for i in range(self.act_dim)]
        return int(np.random.choice(self._mask_or_range(mask)))

    def update(self, obs, action, reward, next_obs, done):
        self._episode_reward += float(reward)

    def get_metrics(self):
        return {"episode_return": round(self._episode_reward, 3)}

    def end_of_episode(self):
        self._episode_reward = 0.0

    def save_model(self, path):
        import torch
        torch.save({"format": "random-v2", "agent_name": "random"}, str(path))

    def load_model(self, path):
        pass
