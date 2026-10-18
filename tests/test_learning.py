"""The learning path and glossary: every reference resolves, every step's
experiment is launchable, and completion is computed correctly."""

import pytest
from fastapi.testclient import TestClient


def _catalogue():
    from rltestbed.core.agents import list_agents
    from rltestbed.core.envs import list_envs

    return {a["name"] for a in list_agents()}, {e["name"]: e for e in list_envs()}


# ------------------------------------------------------------------ glossary
def test_glossary_ids_unique_and_categories_known():
    from rltestbed.glossary import CATEGORIES, GLOSSARY

    ids = [g["id"] for g in GLOSSARY]
    assert len(ids) == len(set(ids))
    assert {g["category"] for g in GLOSSARY} <= set(CATEGORIES)
    assert all(g["definition"] and g["detail"] for g in GLOSSARY)


def test_glossary_references_resolve():
    from rltestbed.glossary import BY_ID, GLOSSARY

    agents, envs = _catalogue()
    for g in GLOSSARY:
        assert set(g["related"]) <= set(BY_ID), (g["id"], set(g["related"]) - set(BY_ID))
        assert set(g["algorithms"]) <= agents, (g["id"], g["algorithms"])
        assert set(g["envs"]) <= set(envs), (g["id"], g["envs"])
        assert g["id"] not in g["related"]


def test_formulas_are_delimited_math():
    from rltestbed.glossary import GLOSSARY

    for g in GLOSSARY:
        if g["formula"]:
            assert g["formula"].startswith("$") and g["formula"].endswith("$"), g["id"]
            assert g["formula"].count("$") % 2 == 0, g["id"]


def test_glossary_endpoint_keeps_the_v3_shape():
    from rltestbed.server.app import create_app

    rows = TestClient(create_app()).get("/api/glossary").json()
    assert all({"term", "definition", "algorithm"} <= set(r) for r in rows)
    assert len(rows) >= 60


# ------------------------------------------------------------------ curriculum
def test_every_concept_is_in_the_glossary():
    from rltestbed.curriculum import iter_steps
    from rltestbed.glossary import BY_ID

    for _, step in iter_steps():
        assert set(step["concepts"]) <= set(BY_ID), (step["id"], set(step["concepts"]) - set(BY_ID))
        assert step["question"] and step["answer"] and step["watch"]


def test_every_step_experiment_validates():
    """Each step launches through experiment files, so it must parse cleanly."""
    from rltestbed.curriculum import _entries, experiment_text, iter_steps
    from rltestbed.experiments import parse

    _, envs = _catalogue()
    checked = 0
    for _, step in iter_steps():
        if any(envs[e["env"]].get("available") is False for _, e in _entries(step)):
            continue  # e.g. the MuJoCo step without MuJoCo installed
        result = parse(experiment_text(step))
        assert result["valid"], (step["id"], result["errors"])
        assert result["summary"]["n_runs"] <= 12, step["id"]  # keep steps laptop-sized
        checked += 1
    assert checked >= 13


def test_step_ids_unique():
    from rltestbed.curriculum import iter_steps

    ids = [s["id"] for _, s in iter_steps()]
    assert len(ids) == len(set(ids))


def _run(agent, env, status="completed", best=None, tags=(), params=None, i=[0]):
    i[0] += 1
    return {"id": f"r{i[0]}", "agent": agent, "env": env, "status": status,
            "best_eval_reward": best, "tags": list(tags), "params": params or {}}


def test_status_counts_seeds_params_and_targets():
    from rltestbed.curriculum import MODULES, step_status

    _, envs = _catalogue()
    step = next(s for m in MODULES for s in m["steps"] if s["id"] == "first-learning")
    runs = [_run("q_learning", "gridworld", best=0.5) for _ in range(3)]
    assert step_status(step, runs, envs)["state"] == "todo"  # target 0.9 not met
    runs[0]["best_eval_reward"] = 0.95
    assert step_status(step, runs, envs)["state"] == "done"

    ucb = next(s for m in MODULES for s in m["steps"] if s["id"] == "ucb-vs-greedy")
    runs = [_run("ucb", "multi_armed_bandit") for _ in range(3)]
    runs += [_run("q_learning", "multi_armed_bandit", params={"gamma": 0.5, "alpha": 0.1}) for _ in range(3)]
    assert step_status(ucb, runs, envs)["state"] == "todo"  # wrong params don't count
    runs += [_run("q_learning", "multi_armed_bandit", status="running", params={"gamma": 0.0, "alpha": 0.1})]
    assert step_status(ucb, runs, envs)["state"] == "running"


def test_unavailable_environments_are_reported():
    from rltestbed.curriculum import MODULES, step_status

    _, envs = _catalogue()
    envs = {**envs, "inverted_pendulum": {**envs["inverted_pendulum"], "available": False}}
    step = next(s for m in MODULES for s in m["steps"] if s["id"] == "mujoco")
    status = step_status(step, [], envs)
    assert status["state"] == "unavailable" and status["missing_envs"] == ["inverted_pendulum"]


def test_curriculum_endpoint_and_legacy_path(tmp_path, monkeypatch):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib

    import rltestbed.config as cfg
    import rltestbed.registry as reg

    importlib.reload(cfg)
    importlib.reload(reg)
    reg.init_db()
    from rltestbed.server.app import create_app

    c = TestClient(create_app())
    data = c.get("/api/curriculum").json()
    assert len(data["modules"]) == 7 and data["done"] == 0 and data["total"] >= 15
    step = data["modules"][0]["steps"][0]
    assert step["status"]["state"] == "todo" and step["experiment_text"].startswith("# Learning path")
    legacy = c.get("/api/learning-path").json()
    assert {"step", "title", "env", "algorithm", "why", "done_when"} <= set(legacy[0])
