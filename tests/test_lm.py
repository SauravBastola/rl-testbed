"""Language-model RL: pretraining, the algorithms actually improving the model,
reward hacking without KL, and the app paths (rollout, evaluate, compatibility)."""

import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def lm_home(tmp_path_factory):
    """One RLTESTBED_HOME for the module, so base models are pretrained once."""
    import os

    home = tmp_path_factory.mktemp("lm")
    old = os.environ.get("RLTESTBED_HOME")
    os.environ["RLTESTBED_HOME"] = str(home)
    import importlib

    import rltestbed.config as cfg

    importlib.reload(cfg)
    yield home
    if old is None:
        os.environ.pop("RLTESTBED_HOME", None)
    else:
        os.environ["RLTESTBED_HOME"] = old
    importlib.reload(cfg)


def _train(home, agent, env, steps, **params):
    from rltestbed.core.runner import create_trainer

    run_dir = Path(home) / "runs" / f"{agent}-{env}-{time.time_ns()}"
    t = create_trainer(run_id="t", run_dir=run_dir, env_name=env, agent_name=agent,
                       params=params, seed=0, episodes=steps, eval_freq=steps)
    return t, t.run()


def test_base_model_is_pretrained_once_and_cached(lm_home):
    from rltestbed.core.lm import base

    t0 = time.time()
    m1 = base.base_model("lm_reverse")
    first = time.time() - t0
    t0 = time.time()
    m2 = base.base_model("lm_reverse")
    assert time.time() - t0 < first / 3  # second call loads from disk
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert (a == b).all()


def test_grpo_sharpens_the_base_model(lm_home):
    trainer, summary = _train(lm_home, "grpo", "lm_addition", 40)
    final = summary["eval_history"][-1]
    assert summary["status"] == "completed"
    assert final["accuracy"] >= 0.75  # base samples are right about half the time
    assert (Path(trainer.run_dir) / "samples.json").exists()


def test_reward_hacking_without_kl(lm_home):
    """With no KL anchor, a length-paying reward model is exploited: reward up, accuracy down."""
    trainer, summary = _train(lm_home, "grpo", "lm_addition_hackable", 50, kl_coef=0.0)
    final = summary["eval_history"][-1]
    assert final["episode_mean"] > 1.6   # more than any honest answer can earn
    assert final["accuracy"] < 0.2       # while being right almost never


def test_dpo_and_reinforce_train(lm_home):
    for agent, env in (("dpo", "lm_sort"), ("lm_reinforce", "lm_sort")):
        _, summary = _train(lm_home, agent, env, 10)
        assert summary["status"] == "completed" and summary["best_eval_reward"] is not None


def test_language_compatibility_is_enforced():
    from rltestbed.capabilities import check_compat
    from rltestbed.core.agents import list_agents
    from rltestbed.core.envs import list_envs

    agents = {a["name"]: a for a in list_agents()}
    envs = {e["name"]: e for e in list_envs()}
    assert check_compat(agents["grpo"], envs["lm_addition"])["compatible"]
    assert not check_compat(agents["dqn"], envs["lm_addition"])["compatible"]
    assert not check_compat(agents["grpo"], envs["cartpole"])["compatible"]
    assert envs["lm_addition"]["lab"] == "language"


def test_visualize_and_evaluate_language_models(lm_home):
    from rltestbed import rollout
    from rltestbed.core.envs import REGISTRY

    r = rollout.run({"env": "lm_reverse", "seed": 0, "episodes": 2})
    assert r["render"]["source"] == "text" and len(r["samples"]) == 16
    sample = r["samples"][0]
    assert sample["tokens"] and all(0 <= t["prob"] <= 1 for t in sample["tokens"])

    trainer, _ = _train(lm_home, "lm_reinforce", "lm_reverse", 5)
    out = rollout.run_language("lm_reverse", REGISTRY["lm_reverse"]["spec"],
                               {"path": str(trainer.best_path), "agent": "lm_reinforce"}, 8, 0)
    assert len(out["samples"]) == 8
