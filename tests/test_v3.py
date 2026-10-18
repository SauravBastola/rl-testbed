"""Tests for v3 behaviour: labs, availability, fixed registrations, seed groups,
aggregation, the job queue, diagnostics, the game adapter and new algorithms."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _env(name):
    from rltestbed.core.envs import list_envs

    return next(e for e in list_envs() if e["name"] == name)


def _agent(name):
    from rltestbed.core.agents import list_agents

    return next(a for a in list_agents() if a["name"] == name)


# ------------------------------------------------------------------ catalogue
def test_every_env_belongs_to_a_known_lab():
    from rltestbed.core.envs import list_envs
    from rltestbed.labs import DIAGNOSTICS, LAB_IDS

    for env in list_envs():
        assert env["lab"] in LAB_IDS + [DIAGNOSTICS], env["name"]
        if env["category"] == "probe":
            assert env["lab"] == DIAGNOSTICS


def test_lab_curricula_reference_real_entries():
    from rltestbed.core.agents import list_agents
    from rltestbed.core.envs import list_envs
    from rltestbed.labs import LABS

    envs = {e["name"] for e in list_envs()}
    agents = {a["name"] for a in list_agents()}
    for lab in LABS:
        for step in lab["curriculum"]:
            assert step["env"] in envs, (lab["id"], step)
            assert step["agent"] in agents, (lab["id"], step)


def test_pendulum_is_continuous_so_sac_trains_and_dqn_is_blocked():
    """v2 dropped action_type for Gymnasium envs, registering Pendulum as discrete."""
    from rltestbed.capabilities import check_compat

    pendulum = _env("pendulum")
    assert pendulum["action_type"] == "continuous"
    assert check_compat(_agent("sac"), pendulum)["compatible"]
    assert not check_compat(_agent("dqn"), pendulum)["compatible"]


def test_pixel_envs_trainable_with_honest_warning():
    from rltestbed.capabilities import check_compat

    v = check_compat(_agent("dqn"), _env("minatar_breakout") | {"available": True})
    assert v["compatible"] and v["level"] == "warning"
    assert any("flattened" in w for w in v["warnings"])


def test_unavailable_env_is_listed_and_blocked():
    from rltestbed.capabilities import check_compat

    fake = _env("cartpole") | {"available": False, "unavailable_reason": "Requires X"}
    v = check_compat(_agent("dqn"), fake)
    assert not v["compatible"] and "Requires X" in v["blockers"]


def test_planned_algorithms_explain_themselves():
    from rltestbed.core.agents import list_agents

    planned = [a for a in list_agents() if a["status"] != "implemented"]
    assert planned
    assert all(a["status_reason"] for a in planned)


def test_theory_lives_in_the_spec():
    assert "$" in _agent("dqn")["docs"]["math"]


# ------------------------------------------------------------------ analysis
def test_aggregate_mean_and_std_across_seeds():
    from rltestbed.analysis import aggregate

    def run(i, seed, rewards):
        return ({"id": f"r{i}", "name": "x", "agent": "dqn", "env": "catch", "seed": seed,
                 "params": {"lr": 1}, "group_id": None},
                [{"episode": e + 1, "reward": r, "mean_reward": r} for e, r in enumerate(rewards)])

    items = [run(0, 0, [0, 2, 4]), run(1, 1, [2, 4, 6]), run(2, 0, [9, 9, 9])]
    items[2][0]["params"] = {"lr": 2}  # a different configuration
    groups = aggregate(items, key="reward", group_by="config")
    assert len(groups) == 2
    pooled = next(g for g in groups if g["n_runs"] == 2)
    assert [p["mean"] for p in pooled["points"]] == [1, 3, 5]
    assert pooled["points"][0]["std"] == pytest.approx(np.std([0, 2], ddof=1))
    single = next(g for g in groups if g["n_runs"] == 1)
    assert single["points"][0]["std"] is None  # one seed has no spread


def test_config_diff_lists_only_differences():
    from rltestbed.analysis import config_diff

    runs = [{"agent": "dqn", "env": "catch", "seed": s, "episodes": 5, "params": {"lr": 1, "g": 0.9}}
            for s in (1, 2)]
    assert set(config_diff(runs)) == {"seed"}


# ------------------------------------------------------------------ API
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RLTESTBED_HOME", str(tmp_path))
    import importlib

    import rltestbed.config as cfg
    import rltestbed.registry as reg

    importlib.reload(cfg)
    importlib.reload(reg)
    reg.init_db()
    from rltestbed.jobs import JOB_MANAGER
    from rltestbed.server.app import create_app

    started = []
    monkeypatch.setattr(JOB_MANAGER, "start", lambda rid: started.append(rid) or True)
    c = TestClient(create_app())
    c.started = started
    return c


def test_labs_endpoint(client):
    labs = client.get("/api/labs").json()
    ids = [lab["id"] for lab in labs]
    assert ids == ["foundations", "classic", "continuous", "perception", "multiagent"]
    assert all("match" not in lab and lab["env_count"] > 0 for lab in labs)


def test_seeds_create_a_group_and_keep_the_old_contract(client):
    r = client.post("/api/runs", json={"agent": "q_learning", "env": "gridworld",
                                       "episodes": 5, "seeds": [1, 2, 3]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == body["run_ids"][0] and len(body["run_ids"]) == 3
    group = client.get(f"/api/groups/{body['group_id']}").json()
    assert sorted(group["seeds"]) == [1, 2, 3]
    runs = client.get(f"/api/runs?group_id={body['group_id']}").json()
    assert {r["lab"] for r in runs} == {"foundations"}


def test_single_run_response_unchanged(client):
    body = client.post("/api/runs", json={"agent": "q_learning", "env": "gridworld",
                                          "episodes": 5}).json()
    assert body["group_id"] is None and body["run_ids"] == [body["run_id"]]


def test_incompatible_launch_is_rejected_with_reason(client):
    r = client.post("/api/runs", json={"agent": "dqn", "env": "pendulum", "episodes": 5})
    assert r.status_code == 422
    assert "continuous" in json.dumps(r.json())


def test_queue_limits_concurrency(client, monkeypatch):
    from rltestbed.jobs import JOB_MANAGER

    monkeypatch.setattr(JOB_MANAGER, "max_concurrent", 1)
    monkeypatch.setattr(JOB_MANAGER, "running_count", lambda: len(client.started))
    client.post("/api/runs", json={"agent": "q_learning", "env": "gridworld", "episodes": 5,
                                   "seeds": [1, 2, 3]})
    statuses = sorted(r["status"] for r in client.get("/api/runs").json())
    assert statuses.count("queued") == 2 and len(client.started) == 1


def test_compare_aggregate_and_history_since(client):
    import rltestbed.config as cfg
    import rltestbed.registry as reg

    ids = []
    for seed in (0, 1):
        rid = reg.create_run("r", "dqn", "catch", seed, 3, {"lr": 1}, {})
        p = cfg.RUNS_DIR / rid / "history.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps({"episode": e, "reward": e + seed}) + "\n" for e in (1, 2, 3)))
        ids.append(rid)
    agg = client.get(f"/api/compare/aggregate?run_ids={','.join(ids)}").json()
    assert len(agg["groups"]) == 1 and agg["groups"][0]["n_runs"] == 2
    assert agg["diff"] == {"seed": [0, 1]}
    assert [h["episode"] for h in client.get(f"/api/runs/{ids[0]}/history?since=2").json()] == [3]


def test_run_patch_and_artifact_download(client):
    import rltestbed.config as cfg
    import rltestbed.registry as reg

    rid = reg.create_run("r", "dqn", "catch", 0, 3, {}, {})
    body = client.patch(f"/api/runs/{rid}", json={"name": "renamed", "tags": ["A", "a", "b"]}).json()
    assert body["name"] == "renamed" and body["tags"] == ["a", "b"]
    (cfg.RUNS_DIR / rid).mkdir(parents=True, exist_ok=True)
    (cfg.RUNS_DIR / rid / "worker.log").write_text("hello")
    assert client.get(f"/api/runs/{rid}/artifacts/worker.log").text == "hello"
    assert client.get(f"/api/runs/{rid}/artifacts/../../etc/passwd").status_code == 404


def test_diagnostics_matrix_and_system(client):
    diag = client.get("/api/diagnostics").json()
    assert len(diag["benchmarks"]) == 5
    cell = diag["benchmarks"][0]["cells"]["dqn"]
    assert cell["verdict"] == "untested" and cell["expected"]
    sysinfo = client.get("/api/system").json()
    assert sysinfo["catalogue"]["agents_implemented"] >= 21


# ------------------------------------------------------------------ envs & agents
def test_game_adapter_scores_from_the_learners_seat():
    pytest.importorskip("pettingzoo")
    from rltestbed.core.envs.adapters import PettingZooSelfPlayAdapter

    env = PettingZooSelfPlayAdapter("tictactoe", seed=0)
    outcomes = []
    for _ in range(60):
        obs, done = env.reset(), False
        while not done:
            legal = [i for i, v in enumerate(obs["action_mask"]) if v]
            obs, r, done, _, info = env.step(legal[0])
        outcomes.append(r)
    # v2 summed both seats' rewards, which is always 0 in a zero-sum game
    assert any(r != 0 for r in outcomes)


def test_random_agent_respects_continuous_bounds():
    from rltestbed.core.agents import create_agent

    a = create_agent("random", 3, 2)
    a.set_action_bounds([-2, -1], [2, 1])
    act = a.get_action({"obs": [0, 0, 0], "action_mask": [1, 1]})
    assert act.shape == (2,) and -2 <= act[0] <= 2 and -1 <= act[1] <= 1


def test_tabular_eval_is_greedy():
    from rltestbed.core.agents import create_agent

    a = create_agent("q_learning", 2, 2, {"epsilon_start": 1.0, "epsilon_end": 1.0})
    a.Q[0] = np.array([0.0, 1.0])
    a.TRAIN = False
    assert {a.get_action({"obs": [1, 0], "action_mask": [1, 1]}) for _ in range(30)} == {1}


@pytest.mark.parametrize("agent,probe,episodes", [
    ("expected_sarsa", "probe3", 200),
    ("double_q", "probe3", 200),
    ("dyna_q", "probe3", 100),
    ("mc_control", "probe3", 200),
    ("ucb", "probe5", 200),
    ("thompson", "probe5", 200),
    ("dqn_per", "probe2", 120),
    # regressions: A2C gathered log-probs by timestep instead of action, and
    # per-episode advantage normalisation erased the baseline on short episodes
    ("a2c", "probe2", 200),
    ("a2c", "probe3", 300),
    ("reinforce_baseline", "probe3", 300),
])
def test_new_algorithms_pass_a_probe(agent, probe, episodes, tmp_path):
    from rltestbed.benchmarks import BENCHMARKS
    from rltestbed.core.trainer import Trainer

    threshold = next(b for b in BENCHMARKS if b["probe"] == probe)["pass_threshold"]
    t = Trainer("t", Path(tmp_path), probe, agent, {}, seed=0, episodes=episodes,
                eval_freq=10, eval_episodes=10, save_best=False, save_final=False)
    assert t.run()["best_eval_reward"] >= threshold
