"""World-model agents: the batched ensemble, what the model learns, termination
versus truncation, the imagination artifact, and both agents end to end."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_batched_ensemble_matches_each_member():
    from rltestbed.core.agents.world_models import EnsembleModel

    torch.manual_seed(0)
    model = EnsembleModel(3, 1, 32, 5)
    s, a = torch.randn(23, 3), torch.randn(23, 1)  # not a multiple of 5: exercises padding
    nxt, r, d = model.step_particles(s, a)
    for i in range(23):
        n_i, r_i, d_i = model.step(i % 5, s[i:i + 1], a[i:i + 1])
        assert torch.allclose(n_i[0], nxt[i], atol=1e-5)
        assert torch.allclose(r_i[0], r[i], atol=1e-5) and torch.allclose(d_i[0], d[i], atol=1e-5)


def _run(tmp_path, agent, env, episodes, **params):
    from rltestbed.core.runner import create_trainer

    run_dir = Path(tmp_path) / f"{agent}-{env}"
    t = create_trainer(run_id="t", run_dir=run_dir, env_name=env, agent_name=agent, params=params,
                       seed=0, episodes=episodes, eval_freq=episodes, eval_episodes=1)
    return t, t.run(), run_dir


def test_time_limits_are_not_learned_as_terminations(tmp_path):
    """Pendulum only ever truncates; the model's 'done' targets must all be zero."""
    trainer, summary, _ = _run(tmp_path, "pets", "pendulum", 2, population=30, cem_iters=1,
                               horizon=5, model_steps=20, warmup_episodes=2)
    assert summary["status"] == "completed"
    assert len(trainer.agent.D) == 400 and sum(trainer.agent.D) == 0


def test_model_learns_pendulum_dynamics_and_publishes_imagination(tmp_path):
    trainer, _, run_dir = _run(tmp_path, "pets", "pendulum", 4, population=30, cem_iters=1,
                               horizon=5, model_steps=300, warmup_episodes=4)
    m = trainer.agent.get_metrics()
    assert m["model_error_1"] < 0.05  # one step ahead: nearly exact
    assert m["model_error_15"] > m["model_error_1"]  # error compounds with horizon
    data = json.loads((run_dir / "imagination.json").read_text())
    assert data["labels"] == ["cos θ", "sin θ", "angular velocity"]
    assert len(data["real"]) == len(data["imagined"]) > 10


def test_pets_plans_on_cartpole_and_dreamer_trains(tmp_path):
    _, summary, _ = _run(tmp_path, "pets", "cartpole", 3, population=60, cem_iters=2, horizon=8,
                         model_steps=50, warmup_episodes=1)
    assert summary["status"] == "completed" and summary["best_eval_reward"] is not None
    trainer, summary, _ = _run(tmp_path, "dreamer_lite", "pendulum", 3, model_steps=50, ac_steps=5,
                               warmup_episodes=1)
    assert summary["status"] == "completed"
    assert "imagined_return" in trainer.agent.get_metrics()


def test_world_model_checkpoint_round_trip(tmp_path):
    from rltestbed.core.agents import create_agent

    trainer, _, _ = _run(tmp_path, "dreamer_lite", "cartpole", 3, model_steps=20, ac_steps=3,
                         warmup_episodes=1)
    path = tmp_path / "m.pth"
    trainer.agent.save_model(path)
    other = create_agent("dreamer_lite", 4, 2, {})
    other.load_model(path)
    other.TRAIN = False
    obs = {"obs": [0.0, 0.1, 0.0, -0.1], "action_mask": [1, 1]}
    trainer.agent.TRAIN = False
    assert other.get_action(obs) == trainer.agent.get_action(obs)


def test_compatibility():
    from rltestbed.capabilities import check_compat
    from rltestbed.core.agents import list_agents
    from rltestbed.core.envs import list_envs

    a = {x["name"]: x for x in list_agents()}
    e = {x["name"]: x for x in list_envs()}
    for agent in ("pets", "dreamer_lite"):
        assert check_compat(a[agent], e["pendulum"])["compatible"]
        assert check_compat(a[agent], e["cartpole"])["compatible"]
        assert not check_compat(a[agent], e["lm_addition"])["compatible"]
