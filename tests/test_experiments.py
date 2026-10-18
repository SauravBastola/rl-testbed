"""Experiment files: parsing, error locations, all-or-nothing launch, export
round trips, the CLI, and CSV exports."""

import csv
import io
import textwrap

import pytest
import yaml
from fastapi.testclient import TestClient

GOOD = textwrap.dedent("""
    version: 1
    name: Smoke
    runs:
      - agent: q_learning
        env: gridworld
        episodes: 5
        seeds: [0, 1]
        params: {alpha: 0.2}
    sweeps:
      - agent: q_learning
        env: gridworld
        episodes: 5
        seeds: [0]
        space:
          alpha: {values: [0.1, 0.5]}
""")


def _errors(text):
    from rltestbed.experiments import parse

    result = parse(text)
    assert not result["valid"]
    return {e["loc"]: e["message"] for e in result["errors"]}


def test_valid_file_is_summarised():
    from rltestbed.experiments import parse

    result = parse(GOOD)
    assert result["valid"], result["errors"]
    assert result["summary"]["n_runs"] == 4  # 2 seeds + 2 sweep configs x 1 seed
    kinds = [e["kind"] for e in result["summary"]["entries"]]
    assert kinds == ["run", "sweep"]


def test_yaml_syntax_errors_report_the_line():
    errs = _errors("runs:\n  - agent: dqn\n    env: [cartpole\n")
    (loc, msg), = errs.items()
    assert loc.startswith("line ") and "YAML" in msg


@pytest.mark.parametrize("text,loc,needle", [
    ("runs:\n  - {agent: dqn, env: cartpole, episods: 5}\n", "runs[0].episods", "Extra inputs"),
    ("sweeps:\n  - {agent: q_learning, env: gridworld, strategey: grid}\n", "sweeps[0].strategey", "Extra"),
    ("runs:\n  - {agent: nope, env: cartpole}\n", "runs[0]", "unknown algorithm"),
    ("runs:\n  - {agent: dqn, env: pendulum}\n", "runs[0]", "continuous"),
    ("runs:\n  - {agent: q_learning, env: gridworld, params: {alpha: 5}}\n", "runs[0]", "alpha"),
    ("sweeps:\n  - {agent: q_learning, env: gridworld}\n", "sweeps[0]", "at least one parameter"),
    ("runs: []\n", "file", "no runs"),
    ("- just\n- a list\n", "file", "mapping"),
    ("version: 2\nruns: []\n", "version", ""),
])
def test_errors_point_at_the_problem(text, loc, needle):
    errs = _errors(text)
    assert loc in errs, errs
    assert needle in errs[loc]


def test_total_run_limit():
    entries = "".join(
        f"  - {{agent: q_learning, env: gridworld, episodes: 1, seeds: {list(range(20))}}}\n"
        for _ in range(26))
    errs = _errors("runs:\n" + entries)
    assert "limit" in errs["file"]


# ------------------------------------------------------------------ server
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


def test_launch_creates_everything_and_tags_runs(client):
    out = client.post("/api/experiments/launch", json={"text": GOOD}).json()
    assert len(out["run_ids"]) == 4 and len(out["sweeps"]) == 1
    runs = client.get("/api/runs").json()
    assert any("exp:Smoke" in r["tags"] for r in runs)


def test_export_keeps_the_experiment_name(client):
    out = client.post("/api/experiments/launch", json={"text": GOOD}).json()
    run_ids = ",".join(out["runs"][0]["run_ids"])
    assert yaml.safe_load(client.get(f"/api/experiments/export?run_ids={run_ids}").text)["name"] == "Smoke"


def test_launch_is_all_or_nothing(client):
    # a valid file plus one incompatible run entry
    bad = GOOD.replace("runs:\n", "runs:\n  - {agent: dqn, env: pendulum, episodes: 5}\n", 1)
    r = client.post("/api/experiments/launch", json={"text": bad})
    assert r.status_code == 422
    assert client.get("/api/runs").json() == []
    assert client.get("/api/sweeps").json() == []


def test_validate_endpoint_returns_errors_as_data(client):
    out = client.post("/api/experiments/validate", json={"text": "runs:\n  - {agent: nope, env: x}\n"}).json()
    assert out["valid"] is False and out["errors"][0]["loc"] == "runs[0]"


def test_seed_group_round_trips_through_export(client):
    from rltestbed.experiments import parse

    created = client.post("/api/runs", json={"agent": "q_learning", "env": "gridworld", "episodes": 7,
                                             "seeds": [3, 4], "params": {"alpha": 0.3}}).json()
    r = client.get(f"/api/experiments/export?run_ids={','.join(created['run_ids'])}")
    assert r.headers["content-disposition"].startswith("attachment")
    doc = yaml.safe_load(r.text)
    (entry,) = doc["runs"]  # two seeds collapse into one entry
    assert entry["seeds"] == [3, 4] and entry["params"]["alpha"] == 0.3 and entry["episodes"] == 7
    assert parse(r.text)["valid"]


def test_sweep_round_trips_through_export(client):
    from rltestbed.experiments import parse

    body = {"agent": "q_learning", "env": "gridworld", "episodes": 5, "seeds": [0, 1],
            "space": {"alpha": {"values": [0.1, 0.5]}}, "name": "alpha sweep"}
    sid = client.post("/api/sweeps", json=body).json()["sweep_id"]
    text = client.get(f"/api/experiments/export?sweep_id={sid}").text
    (sweep,) = yaml.safe_load(text)["sweeps"]
    assert sweep["space"] == body["space"] and sweep["seeds"] == [0, 1]
    result = parse(text)
    assert result["valid"] and result["summary"]["n_runs"] == 4


def test_examples_parse_when_their_environments_are_installed(client):
    from rltestbed.core.envs import list_envs
    from rltestbed.experiments import parse

    available = {e["name"] for e in list_envs() if e.get("available")}
    examples = client.get("/api/experiments/examples").json()
    assert len(examples) == 5
    for ex in examples:
        envs = {r["env"] for r in yaml.safe_load(ex["text"])["runs"]}
        if envs <= available:
            assert parse(ex["text"])["valid"], ex["id"]


def test_history_and_comparison_csv(client):
    import json

    import rltestbed.config as cfg
    import rltestbed.registry as reg

    rid = reg.create_run("r", "dqn", "catch", 0, 3, {}, {})
    path = cfg.RUNS_DIR / rid / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"episode": 1, "reward": 1.0, "metric_loss": 0.5},
        {"episode": 2, "reward": 2.0, "eval": {"episode_mean": 3.0, "episode_std": 0.1}},
    ]))
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/runs/{rid}/history.csv").text)))
    assert rows[0]["metric_loss"] == "0.5" and rows[1]["eval_mean"] == "3.0"
    cmp = client.get(f"/api/compare/aggregate?run_ids={rid}&format=csv")
    assert cmp.headers["content-type"].startswith("text/csv")
    assert next(csv.reader(io.StringIO(cmp.text)))[:3] == ["group", "n_runs", "episode"]


# ------------------------------------------------------------------ CLI
def test_cli_dry_run_and_export(client, tmp_path):
    from typer.testing import CliRunner

    from rltestbed.cli import app as cli

    runner = CliRunner()
    good = tmp_path / "good.yaml"
    good.write_text(GOOD)
    res = runner.invoke(cli, ["run", str(good), "--dry-run"])
    assert res.exit_code == 0 and "4 runs" in res.output and "nothing was created" in res.output

    bad = tmp_path / "bad.yaml"
    bad.write_text("runs:\n  - {agent: dqn, env: cartpole, episods: 5}\n")
    res = runner.invoke(cli, ["run", str(bad), "--dry-run"])
    assert res.exit_code == 1 and "runs[0].episods" in res.output

    created = client.post("/api/runs", json={"agent": "q_learning", "env": "gridworld", "episodes": 5}).json()
    out = tmp_path / "exported.yaml"
    res = runner.invoke(cli, ["export", created["run_id"], "-o", str(out)])
    assert res.exit_code == 0 and yaml.safe_load(out.read_text())["runs"][0]["agent"] == "q_learning"
