"""The real execution path: `rltestbed train-worker --run-id` in a subprocess,
for both training loops (step-by-step and language-model)."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("agent,env,episodes", [("q_learning", "gridworld", 5), ("grpo", "lm_reverse", 3)])
def test_train_worker_completes(tmp_path, agent, env, episodes):
    env_vars = {**os.environ, "RLTESTBED_HOME": str(tmp_path)}
    code = (
        "import json, rltestbed.registry as reg\n"
        f"rid = reg.create_run('t', '{agent}', '{env}', 0, {episodes}, {{}}, "
        f"{{'episodes': {episodes}, 'seed': 0, 'eval_freq': 1, 'eval_episodes': 2}})\n"
        "print(rid)\n"
    )
    rid = subprocess.run([sys.executable, "-c", code], env=env_vars, capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()[-1]
    proc = subprocess.run([sys.executable, "-m", "rltestbed.cli", "train-worker", "--run-id", rid],
                          env=env_vars, capture_output=True, text=True, timeout=240)
    log = (tmp_path / "runs" / rid / "worker.log").read_text()
    assert proc.returncode == 0, log[-2000:] + proc.stderr[-2000:]
    history = [json.loads(line) for line in (tmp_path / "runs" / rid / "history.jsonl").read_text().splitlines()]
    assert len(history) == episodes and any("eval" in h for h in history)
    status = subprocess.run(
        [sys.executable, "-c", f"import rltestbed.registry as reg; print(reg.get_run('{rid}')['status'])"],
        env=env_vars, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    assert status == "completed"
