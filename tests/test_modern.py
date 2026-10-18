"""v3.7: time limits are not terminals, SimBa networks, CrossQ, DAPO."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("agent,obs_dim,act_dim,action", [
    ("sac", 3, 1, np.array([0.5], dtype=np.float32)),
    ("td3", 3, 1, np.array([0.5], dtype=np.float32)),
    ("dqn", 4, 2, 1),
])
def test_time_limits_are_stored_as_non_terminal(agent, obs_dim, act_dim, action):
    """Before v3.7 every time limit (e.g. Pendulum's 200 steps) was stored as a
    true terminal, so these agents learned the last state is worth nothing."""
    from rltestbed.core.agents import create_agent

    a = create_agent(agent, obs_dim, act_dim, {})
    if agent != "dqn":
        a.set_action_bounds(-np.ones(1), np.ones(1))
    obs = [0.0] * obs_dim
    for truncated, expected in ((True, 0.0), (False, 1.0)):
        a.last_step_truncated = truncated
        a.update(obs, action, 1.0, obs, True)
        if agent == "dqn":
            stored = float(a.buffer[-1][-1])  # deque of (s, a, r, s2, done)
        else:
            stored = float(a.buffer.dones[(a.buffer.ptr - 1) % a.buffer.max_size])
        assert stored == expected


def test_default_network_layout_is_unchanged():
    """Old checkpoints must still load: the MLP parameter names are as before."""
    from rltestbed.core.agents import create_agent

    sac = create_agent("sac", 3, 1, {})
    keys = set(sac.actor.state_dict()) | {f"critic.{k}" for k in sac.critic.state_dict()}
    assert {"trunk.0.weight", "trunk.2.weight", "mu.weight", "critic.q1.net.0.weight"} <= keys


def test_simba_networks_build_and_run():
    from rltestbed.core.agents import create_agent

    for name in ("sac", "td3"):
        a = create_agent(name, 3, 1, {"network": "simba"})
        a.set_action_bounds(-2 * np.ones(1), 2 * np.ones(1))
        out = a.get_action([0.1, 0.2, 0.3])
        assert np.all(np.abs(out) <= 2.0)
        assert any(isinstance(m, torch.nn.LayerNorm) for m in a.actor.modules())


def test_crossq_has_no_target_network_and_trains(tmp_path):
    from rltestbed.core.runner import create_trainer

    t = create_trainer(run_id="c", run_dir=Path(tmp_path) / "c", env_name="pendulum",
                       agent_name="crossq", params={"warmup_steps": 64, "batch_size": 32,
                                                    "hidden_size": 64},
                       seed=0, episodes=2, eval_freq=2, eval_episodes=1)
    assert t.agent.critic_target is t.agent.critic
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in t.agent.critic.modules())
    summary = t.run()
    assert summary["status"] == "completed" and t.agent.updates > 100
    assert "alpha" in t.agent.get_metrics()


def test_dapo_preset_and_dynamic_sampling(tmp_path, monkeypatch):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib

    import rltestbed.config as cfg

    importlib.reload(cfg)
    from rltestbed.core.agents import create_agent
    from rltestbed.core.lm.base import base_model
    from rltestbed.core.lm.tasks import TASKS

    preset = next(p for p in create_agent("grpo", 1, 1, {}).SPEC["presets"] if p["name"] == "dapo")
    assert preset["overrides"]["clip_eps_high"] > 0.2 and preset["overrides"]["kl_coef"] == 0.0
    agent = create_agent("grpo", 1, 1, {**preset["overrides"], "prompts_per_step": 4, "group_size": 4})
    agent.attach(base_model("lm_reverse"))
    m = agent.train_step(TASKS["lm_reverse"], random.Random(0))
    assert "resample_rounds" in m and 0.0 <= m["clip_fraction"] <= 1.0
    importlib.reload(cfg)


@pytest.mark.parametrize("agent", ["sac", "td3", "ddpg", "crossq"])
def test_replay_stores_raw_observations_and_normalises_at_sampling(agent):
    """Regression: v3.6 and earlier stored observations normalised with the
    statistics of the moment they were stored. The earliest ones (statistics
    from a few steps) stayed inconsistent with everything after, and SAC, TD3,
    DDPG and CrossQ failed to learn Pendulum. Measured: SAC's mean return over
    episodes 31-40 went from about -1100 to about -130 with the fix."""
    from rltestbed.core.agents import create_agent

    a = create_agent(agent, 2, 1, {"batch_size": 8, "warmup_steps": 10**6})
    a.set_action_bounds(-np.ones(1), np.ones(1))
    rng = np.random.default_rng(0)
    first = np.array([100.0, -50.0], dtype=np.float32)  # far from later data
    a.last_step_truncated = False
    a.update(first, np.zeros(1, np.float32), 0.0, first, False)
    for _ in range(500):
        s = rng.normal(size=2).astype(np.float32)
        a.update(s, np.zeros(1, np.float32), 0.0, s, False)
    # the buffer holds the raw observation
    np.testing.assert_allclose(a.buffer.obs[0], first)
    # sampling normalises with the statistics of now, not of then
    a.buffer.size = 1  # make the first transition the only one to sample
    obs, *_ = a._sample_batch()
    np.testing.assert_allclose(obs[0].numpy(), a.normalizer.normalize(first), rtol=1e-5)
