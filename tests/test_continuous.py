"""Continuous-action learners: the probe6 diagnostic, Gaussian PPO and SAC's critic."""

import warnings
from pathlib import Path

import numpy as np
import pytest


def _diag_params(name):
    from rltestbed.benchmarks import diagnostic_params
    from rltestbed.capabilities import validate_params
    from rltestbed.core.agents import list_agents

    spec = next(a for a in list_agents() if a["name"] == name)
    return validate_params(spec, diagnostic_params(spec))


@pytest.mark.parametrize("agent", ["sac", "td3", "ddpg", "ppo_continuous"])
def test_continuous_learners_pass_probe6(agent, tmp_path):
    from rltestbed.core.trainer import Trainer

    t = Trainer("t", Path(tmp_path), "probe6", agent, _diag_params(agent), seed=0,
                episodes=300, eval_freq=10, eval_episodes=10, save_best=False, save_final=False)
    assert t.run()["best_eval_reward"] >= 0.9


def test_probe6_is_continuous_and_blocks_discrete_agents():
    from rltestbed.capabilities import check_compat
    from rltestbed.core.agents import list_agents
    from rltestbed.core.envs import list_envs

    env = next(e for e in list_envs() if e["name"] == "probe6")
    agents = {a["name"]: a for a in list_agents()}
    assert env["action_type"] == "continuous" and env["lab"] == "diagnostics"
    assert check_compat(agents["ppo_continuous"], env)["compatible"]
    assert not check_compat(agents["ppo"], env)["compatible"]


def test_sac_critic_target_has_batch_shape():
    """Regression: log-probs shaped (B, 1) broadcast the target to (B, B)."""
    from rltestbed.core.agents import create_agent

    agent = create_agent("sac", 3, 1, {"batch_size": 16, "warmup_steps": 0})
    agent.set_action_bounds([-2.0], [2.0])
    rng = np.random.default_rng(0)
    for _ in range(32):
        agent.buffer.add(rng.normal(size=3), rng.uniform(-1, 1, size=1), rng.normal(),
                         rng.normal(size=3), False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a broadcasting mse_loss raises here
        agent._learn()


def test_gaussian_ppo_rescales_and_round_trips(tmp_path):
    from rltestbed.core.agents import create_agent

    agent = create_agent("ppo_continuous", 3, 2, {})
    agent.set_action_bounds([-2.0, 0.0], [2.0, 10.0])
    agent.TRAIN = False
    act = agent.get_action({"obs": [0.1, 0.2, 0.3], "action_mask": [1, 1]})
    assert act.shape == (2,) and -2 <= act[0] <= 2 and 0 <= act[1] <= 10
    path = tmp_path / "m.pth"
    agent.save_model(path)
    other = create_agent("ppo_continuous", 3, 2, {})
    other.load_model(path)
    other.TRAIN = False
    np.testing.assert_allclose(other.get_action({"obs": [0.1, 0.2, 0.3], "action_mask": [1, 1]}), act, rtol=1e-5)
