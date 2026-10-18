"""Validate and create runs and sweeps.

The single implementation behind ``POST /api/runs``, ``POST /api/sweeps``,
experiment files and the CLI. Each operation has a ``check_*`` step that only
validates (and raises :class:`LaunchError` with every reason) and a ``create_*``
step that writes to the registry and queues the work. Experiment files check
every entry before creating any, so a file either launches completely or not
at all.
"""

from typing import Any, Dict, List, Optional, Tuple

from rltestbed import sweeps as sweeps_mod
from rltestbed.capabilities import check_compat, validate_params
from rltestbed.core.agents import list_agents
from rltestbed.core.envs import list_envs
from rltestbed.jobs import JOB_MANAGER
from rltestbed.registry import create_run, create_sweep, new_group_id

RUN_CONFIG_KEYS = (
    "episodes", "seed", "eval_freq", "eval_episodes", "checkpoint_freq", "max_steps",
    "save_best", "save_final", "auto_eval", "early_stopping_patience", "log_tensorboard",
    "log_wandb", "preset", "mode",
)


class LaunchError(Exception):
    """A request that cannot be launched. ``reasons`` lists every problem."""

    def __init__(self, message: str, reasons: Optional[List[str]] = None, status: int = 422):
        super().__init__(message)
        self.message = message
        self.reasons = reasons or []
        self.status = status

    def detail(self) -> Any:
        return {"message": self.message, "reasons": self.reasons} if self.reasons else self.message

    def __str__(self) -> str:
        return f"{self.message}: {'; '.join(self.reasons)}" if self.reasons else self.message


def _specs(agent_name: str, env_name: str):
    agent = next((a for a in list_agents() if a["name"] == agent_name), None)
    env = next((e for e in list_envs() if e["name"] == env_name), None)
    if not agent:
        raise LaunchError(f"unknown algorithm '{agent_name}'")
    if not env:
        raise LaunchError(f"unknown environment '{env_name}'")
    compat = check_compat(agent, env)
    if not compat["compatible"]:
        raise LaunchError("incompatible training configuration", compat["blockers"])
    return agent, env, compat


# ---------------------------------------------------------------------- runs
def check_run(payload: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict]:
    """Validate a run request (a ``RunCreateRequest.model_dump()``).

    Returns (agent spec, env spec, typed params, compatibility verdict).
    """
    agent, env, compat = _specs(payload["agent"], payload["env"])
    try:
        typed = validate_params(agent, payload.get("params") or {})
    except ValueError as exc:
        raise LaunchError(str(exc)) from exc
    device = payload.get("device", "cpu")
    if device in ("cuda", "mps"):
        import torch

        ok = torch.cuda.is_available() if device == "cuda" else (
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
        if not ok:
            raise LaunchError(f"{device.upper()} was selected but is not available")
    return agent, env, typed, compat


def n_seeds(payload: Dict[str, Any]) -> int:
    return len(payload.get("seeds") or [payload.get("seed", 0)])


def create_runs(payload: Dict[str, Any], enqueue: bool = True) -> Dict[str, Any]:
    """Create one run per seed (a seed group when there are several) and queue them."""
    agent, env, typed, compat = check_run(payload)
    name = payload.get("name") or f"{agent['name']}-{env['name']}"
    config = {k: payload[k] for k in RUN_CONFIG_KEYS if k in payload}
    config["compatibility"] = compat
    seeds = payload.get("seeds") or [payload["seed"]]
    group_id = new_group_id() if len(seeds) > 1 else None
    run_ids = []
    for seed in seeds:
        run_name = name if len(seeds) == 1 else f"{name} · s{seed}"
        rid = create_run(run_name, agent["name"], env["name"], seed, payload["episodes"], typed,
                         {**config, "seed": seed}, tags=payload.get("tags") or [],
                         notes=payload.get("description", ""), device=payload.get("device", "cpu"),
                         group_id=group_id)
        run_ids.append(rid)
    if enqueue:
        for rid in run_ids:
            JOB_MANAGER.enqueue(rid)
    return {"run_id": run_ids[0], "run_ids": run_ids, "group_id": group_id}


# -------------------------------------------------------------------- sweeps
def check_sweep(body: Dict[str, Any]):
    """Validate a sweep request; returns (agent, env, configs, seeds, compat)."""
    agent, env, compat = _specs(body.get("agent"), body.get("env"))
    try:
        configs, seeds = sweeps_mod.plan(agent, body)
    except sweeps_mod.SweepError as exc:
        raise LaunchError("invalid sweep", exc.errors) from exc
    return agent, env, configs, seeds, compat


def create_sweep_runs(body: Dict[str, Any], enqueue: bool = True) -> Dict[str, Any]:
    agent, env, configs, seeds, compat = check_sweep(body)
    episodes = int(body.get("episodes") or 200)
    if not 1 <= episodes <= 1_000_000:
        raise LaunchError("episodes must be between 1 and 1,000,000")
    name = (body.get("name") or f"{agent['name']}-{env['name']} sweep").strip()[:160]
    strategy = body.get("strategy", "grid")
    sid = create_sweep(name, agent["name"], env["name"], episodes, strategy, body)
    base = body.get("base_params") or {}
    config = {
        "episodes": episodes, "eval_freq": int(body.get("eval_freq") or 25),
        "eval_episodes": int(body.get("eval_episodes") or 10), "checkpoint_freq": 1000,
        "save_best": True, "save_final": False, "compatibility": compat,
    }
    run_ids = []
    for i, swept in enumerate(configs):
        params = validate_params(agent, {**base, **swept})
        gid = new_group_id()
        for seed in seeds:
            rid = create_run(f"{name} · c{i + 1} · s{seed}", agent["name"], env["name"], seed,
                             episodes, params, {**config, "seed": seed}, tags=["sweep"],
                             group_id=gid, sweep_id=sid)
            run_ids.append(rid)
    # enqueue after creation so the queue sees the whole sweep in order
    if enqueue:
        for rid in run_ids:
            JOB_MANAGER.enqueue(rid)
    return {"sweep_id": sid, "run_ids": run_ids, "n_configs": len(configs)}
