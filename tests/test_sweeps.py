"""Hyperparameter sweeps: expansion, validation, API, ranking and effects."""

import json

import pytest
from fastapi.testclient import TestClient


def _agent(name):
    from rltestbed.core.agents import list_agents

    return next(a for a in list_agents() if a["name"] == name)


# ------------------------------------------------------------------ expansion
def test_grid_is_the_cartesian_product():
    from rltestbed.sweeps import expand

    configs = expand(_agent("q_learning"),
                     {"alpha": {"values": [0.1, 0.5]}, "gamma": {"values": [0.9, 0.95, 0.99]}})
    assert len(configs) == 6
    assert {"alpha": 0.5, "gamma": 0.95} in configs


def test_random_search_respects_log_ranges_and_is_reproducible():
    from rltestbed.sweeps import expand

    space = {"lr": {"min": 1e-5, "max": 1e-2, "log": True}, "batch_size": {"values": [32, 64]}}
    a = expand(_agent("dqn"), space, "random", n_trials=20, seed=3)
    b = expand(_agent("dqn"), space, "random", n_trials=20, seed=3)
    assert a == b
    assert all(1e-5 <= c["lr"] <= 1e-2 and c["batch_size"] in (32, 64) for c in a)
    # log-uniform: a good share of samples fall below the arithmetic midpoint
    assert sum(c["lr"] < 5e-3 for c in a) > len(a) / 2


def test_int_parameters_are_cast():
    from rltestbed.sweeps import expand

    cfgs = expand(_agent("dqn"), {"hidden_size": {"min": 32, "max": 128}}, "random", n_trials=5)
    assert all(isinstance(c["hidden_size"], int) for c in cfgs)


@pytest.mark.parametrize("space,strategy,needle", [
    ({}, "grid", "at least one parameter"),
    ({"nope": {"values": [1]}}, "grid", "no parameter 'nope'"),
    ({"alpha": {"min": 0.1, "max": 0.5}}, "grid", "needs a list"),
    ({"alpha": {"min": 0.5, "max": 0.1}}, "random", "min must be below max"),
    ({"alpha": {"values": [5.0]}}, "grid", "alpha"),  # outside the ParamSpec range
])
def test_invalid_spaces_explain_themselves(space, strategy, needle):
    from rltestbed.sweeps import SweepError, expand

    with pytest.raises(SweepError) as err:
        expand(_agent("q_learning"), space, strategy)
    assert needle in str(err.value)


def test_oversized_sweeps_are_refused_before_expansion():
    from rltestbed.sweeps import SweepError, plan

    space = {n: {"values": list(range(1, 11))} for n in ("batch_size", "hidden_size", "target_update_freq")}
    with pytest.raises(SweepError, match="limit"):
        plan(_agent("dqn"), {"space": space, "seeds": [0]})


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

    monkeypatch.setattr(JOB_MANAGER, "start", lambda rid: True)
    monkeypatch.setattr(JOB_MANAGER, "running_count", lambda: 0)
    return TestClient(create_app())


SWEEP = {"agent": "q_learning", "env": "gridworld", "episodes": 5, "seeds": [0, 1],
         "strategy": "grid", "space": {"alpha": {"values": [0.1, 0.5]}}}


def test_preview_reports_counts_and_errors(client):
    ok = client.post("/api/sweeps/preview", json=SWEEP).json()
    assert ok["valid"] and ok["n_configs"] == 2 and ok["n_runs"] == 4
    bad = client.post("/api/sweeps/preview", json={**SWEEP, "space": {"nope": {"values": [1]}}}).json()
    assert not bad["valid"] and "nope" in bad["errors"][0]


def test_incompatible_sweep_is_rejected(client):
    r = client.post("/api/sweeps", json={**SWEEP, "agent": "dqn", "env": "pendulum",
                                         "space": {"lr": {"values": [1e-3]}}})
    assert r.status_code == 422


def _write_history(rid, rewards):
    import rltestbed.config as cfg

    path = cfg.RUNS_DIR / rid / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"episode": i + 1, "reward": r}) + "\n" for i, r in enumerate(rewards)))


def test_sweep_ranks_configs_and_reports_effects(client):
    created = client.post("/api/sweeps", json=SWEEP).json()
    assert created["n_configs"] == 2 and len(created["run_ids"]) == 4
    runs = client.get(f"/api/runs?sweep_id={created['sweep_id']}").json()
    assert len({r["group_id"] for r in runs}) == 2  # one seed group per configuration
    for r in runs:  # alpha 0.5 scores higher
        _write_history(r["id"], [1.0] * 10 if r["params"]["alpha"] == 0.5 else [0.2] * 10)

    sw = client.get(f"/api/sweeps/{created['sweep_id']}").json()
    assert [c["params"]["alpha"] for c in sw["configs"]] == [0.5, 0.1]
    assert sw["configs"][0]["rank"] == 1 and sw["configs"][0]["n_runs"] == 2
    effect = {row["value"]: row["mean"] for row in sw["effects"]["alpha"]["values"]}
    assert effect[0.5] == pytest.approx(1.0) and effect[0.1] == pytest.approx(0.2)
    listed = client.get("/api/sweeps?lab=foundations").json()
    assert listed[0]["id"] == created["sweep_id"] and listed[0]["n_runs"] == 4


def test_stop_and_delete_sweep(client):
    import rltestbed.registry as reg

    created = client.post("/api/sweeps", json=SWEEP).json()
    for rid in created["run_ids"]:
        reg.update_run(rid, status="queued")
    assert client.post(f"/api/sweeps/{created['sweep_id']}/stop").json()["stopped"] == 4
    assert {r["status"] for r in client.get(f"/api/runs?sweep_id={created['sweep_id']}").json()} == {"stopped"}
    out = client.delete(f"/api/sweeps/{created['sweep_id']}").json()
    assert out["runs_deleted"] == 4
    assert client.get(f"/api/sweeps/{created['sweep_id']}").status_code == 404
    assert client.get(f"/api/runs?sweep_id={created['sweep_id']}").json() == []
