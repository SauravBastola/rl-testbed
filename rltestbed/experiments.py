"""Experiment files: runs and sweeps described in YAML, so results can be
reproduced, reviewed and versioned alongside the code.

Format (every field except ``agent`` and ``env`` is optional)::

    version: 1
    name: CartPole baselines
    description: DQN vs PPO, three seeds each.
    runs:
      - agent: dqn
        env: cartpole
        episodes: 300
        seeds: [0, 1, 2]
        params: {lr: 0.0005}
    sweeps:
      - agent: q_learning
        env: cliff_walking
        episodes: 150
        seeds: [0, 1]
        strategy: grid
        space:
          alpha: {values: [0.05, 0.2, 0.5]}

``runs`` entries accept the fields of ``POST /api/runs``; ``sweeps`` entries the
fields of ``POST /api/sweeps``. Unknown keys are errors, so a typo cannot be
silently ignored. A file is validated completely before anything is created:
it launches entirely or not at all.

Exports go the other way: any run, set of runs or sweep becomes a file that
recreates it. Hyperparameters are written in full (not only the non-defaults),
so a file still means the same thing after a default changes.
"""

import json
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rltestbed import launch
from rltestbed.schemas import RunCreateRequest, SweepCreateRequest

MAX_TOTAL_RUNS = 500
HEADER = (
    "# RL Testbed experiment file.\n"
    "# Validate: rltestbed run {file} --dry-run    Launch: rltestbed run {file}\n"
    "# Or paste it into the Experiments page of the web app.\n"
)


class ExperimentFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    name: str = Field(default="Untitled experiment", min_length=1, max_length=160)
    description: str = Field(default="", max_length=4000)
    runs: List[RunCreateRequest] = Field(default_factory=list)
    sweeps: List[SweepCreateRequest] = Field(default_factory=list)


def _loc(parts) -> str:
    out = ""
    for p in parts:
        out += f"[{p}]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out or "file"


def parse(text: str) -> Dict[str, Any]:
    """Parse and validate an experiment file.

    Returns ``{"valid", "errors": [{"loc", "message"}], "experiment", "summary"}``.
    ``experiment`` holds normalised entries (defaults filled in) when valid.
    """
    errors: List[Dict[str, str]] = []

    def fail():
        return {"valid": False, "errors": errors, "experiment": None, "summary": None}

    try:
        data = yaml.safe_load(text) if text and text.strip() else None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "file"
        errors.append({"loc": where, "message": f"Not valid YAML: {getattr(exc, 'problem', exc)}"})
        return fail()
    if not isinstance(data, dict):
        errors.append({"loc": "file", "message": "Expected a mapping with 'runs' and/or 'sweeps'."})
        return fail()

    try:
        exp = ExperimentFile(**data)
    except ValidationError as exc:
        for e in exc.errors():
            errors.append({"loc": _loc(e["loc"]), "message": e["msg"]})
        return fail()

    if not exp.runs and not exp.sweeps:
        errors.append({"loc": "file", "message": "The file has no runs and no sweeps."})
        return fail()

    entries, total = [], 0
    for i, run in enumerate(exp.runs):
        payload = run.model_dump()
        try:
            launch.check_run(payload)
        except launch.LaunchError as exc:
            errors.append({"loc": f"runs[{i}]", "message": str(exc)})
            continue
        n = launch.n_seeds(payload)
        total += n
        entries.append({"kind": "run", "index": i, "agent": run.agent, "env": run.env,
                        "episodes": run.episodes, "n_runs": n})
    for i, sweep in enumerate(exp.sweeps):
        body = sweep.model_dump()
        try:
            _, _, configs, seeds, _ = launch.check_sweep(body)
        except launch.LaunchError as exc:
            errors.append({"loc": f"sweeps[{i}]", "message": str(exc)})
            continue
        n = len(configs) * len(seeds)
        total += n
        entries.append({"kind": "sweep", "index": i, "agent": sweep.agent, "env": sweep.env,
                        "episodes": sweep.episodes, "n_configs": len(configs), "n_runs": n})
    if total > MAX_TOTAL_RUNS:
        errors.append({"loc": "file",
                       "message": f"The file creates {total} runs; the limit is {MAX_TOTAL_RUNS}."})
    if errors:
        return fail()
    return {
        "valid": True,
        "errors": [],
        "experiment": exp,
        "summary": {"name": exp.name, "description": exp.description, "entries": entries,
                    "n_runs": total, "n_entries": len(entries)},
    }


def launch_file(text: str, enqueue: bool = True) -> Dict[str, Any]:
    """Validate everything, then create every run and sweep in the file."""
    result = parse(text)
    if not result["valid"]:
        raise launch.LaunchError("invalid experiment file",
                                 [f"{e['loc']}: {e['message']}" for e in result["errors"]])
    exp: ExperimentFile = result["experiment"]
    tag = f"exp:{exp.name}"[:40]
    out = {"name": exp.name, "runs": [], "sweeps": [], "run_ids": []}
    for run in exp.runs:
        payload = run.model_dump()
        payload["tags"] = list(dict.fromkeys((payload.get("tags") or []) + [tag]))[:30]
        created = launch.create_runs(payload, enqueue=enqueue)
        out["runs"].append(created)
        out["run_ids"] += created["run_ids"]
    for sweep in exp.sweeps:
        created = launch.create_sweep_runs(sweep.model_dump(), enqueue=enqueue)
        out["sweeps"].append(created)
        out["run_ids"] += created["run_ids"]
    out["summary"] = result["summary"]
    return out


# -------------------------------------------------------------------- export
def _run_key(run: Dict[str, Any]) -> str:
    cfg = run.get("config") or {}
    return json.dumps([run["agent"], run["env"], run["episodes"], run.get("params") or {},
                       cfg.get("eval_freq"), cfg.get("eval_episodes"), cfg.get("max_steps"),
                       run.get("device", "cpu")], sort_keys=True)


def default_name(runs: List[Dict[str, Any]], sweeps: List[Dict[str, Any]]) -> str:
    """The original experiment's name if every run came from it, else a sensible one."""
    if sweeps:
        return sweeps[0]["name"]
    names = {t[4:] for r in runs for t in (r.get("tags") or []) if t.startswith("exp:")}
    if len(names) == 1 and all(any(t.startswith("exp:") for t in r.get("tags") or []) for r in runs):
        return names.pop()
    return runs[0]["name"].split(" · ")[0]


def run_entries(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs that differ only by seed collapse into one entry with a seed list."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault(_run_key(r), []).append(r)
    entries = []
    for members in groups.values():
        head = members[0]
        cfg = head.get("config") or {}
        seeds = sorted({int(m["seed"]) for m in members})
        entry: Dict[str, Any] = {"agent": head["agent"], "env": head["env"],
                                 "episodes": int(head["episodes"])}
        name = head["name"].split(" · s")[0]
        if name and name != f"{head['agent']}-{head['env']}":
            entry["name"] = name
        if len(seeds) == 1:
            entry["seed"] = seeds[0]
        else:
            entry["seeds"] = seeds
        if head.get("params"):
            entry["params"] = head["params"]
        for key in ("eval_freq", "eval_episodes", "max_steps"):
            if cfg.get(key) is not None:
                entry[key] = cfg[key]
        if head.get("device", "cpu") != "cpu":
            entry["device"] = head["device"]
        tags = [t for t in (head.get("tags") or []) if not t.startswith("exp:") and t != "sweep"]
        if tags:
            entry["tags"] = tags
        entries.append(entry)
    return entries


def sweep_entry(sweep: Dict[str, Any]) -> Dict[str, Any]:
    req = dict(sweep["request"])
    entry = {"name": sweep["name"], "agent": sweep["agent"], "env": sweep["env"],
             "episodes": int(sweep["episodes"])}
    for key in ("strategy", "space", "n_trials", "sample_seed", "base_params", "seeds",
                "eval_freq", "eval_episodes"):
        if key in req and req[key] not in (None, {}, []):
            if key == "n_trials" and req.get("strategy", "grid") == "grid":
                continue
            entry[key] = req[key]
    return entry


def dump(name: str, runs: Optional[List[Dict]] = None, sweeps: Optional[List[Dict]] = None,
         description: str = "", filename: str = "experiment.yaml") -> str:
    """Serialise an experiment as YAML that `parse` accepts."""
    doc: Dict[str, Any] = {"version": 1, "name": name}
    if description:
        doc["description"] = description
    if runs:
        doc["runs"] = run_entries(runs)
    if sweeps:
        doc["sweeps"] = [sweep_entry(s) for s in sweeps]
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=None,
                          width=100)
    return HEADER.format(file=filename) + body


def examples(labs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """One starter file per lab, built from its curriculum."""
    out = []
    for lab in labs:
        runs = [{"agent": s["agent"], "env": s["env"], "episodes": s["episodes"],
                 "seed": 0, "params": {}, "name": s["title"], "config": {}} for s in lab["curriculum"]]
        text = dump(f"{lab['label']} curriculum", runs=runs,
                    description=f"Every step of the {lab['label']} lab curriculum, one seed each.",
                    filename=f"{lab['id']}.yaml")
        out.append({"id": lab["id"], "label": f"{lab['label']} curriculum", "text": text})
    return out
