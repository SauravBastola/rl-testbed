"""Capability & compatibility system.

Single source of truth deciding which algorithm can train on which environment.
Both the API (pre-launch validation) and the UI (compatibility panels) call
`check_compat`; there is no duplicated compatibility logic anywhere else.

Observation types:
    onehot  - one-hot index vector (tabular-friendly)
    vector  - flat float vector (CartPole, LunarLander, Catch, MinAtar-flattened)
    image   - multi-channel pixel grid (MinAtar; flattened for the current MLPs)
Action types: discrete | continuous
"""

from typing import Any, Dict, List


def validate_params(agent: Dict[str, Any], supplied: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve and validate dynamic hyperparameters from an algorithm schema."""
    specs = {item["name"]: item for item in agent.get("params", [])}
    unknown = sorted(set(supplied) - set(specs))
    if unknown:
        raise ValueError(f"unknown hyperparameters: {unknown}")

    resolved: Dict[str, Any] = {}
    for name, spec in specs.items():
        value = supplied.get(name, spec.get("default"))
        kind = spec.get("type")
        try:
            if kind == "bool":
                if isinstance(value, bool):
                    pass
                elif isinstance(value, str) and value.lower() in {"true", "false"}:
                    value = value.lower() == "true"
                else:
                    raise ValueError("must be true or false")
            elif kind == "int":
                if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                    raise ValueError("must be an integer")
                value = int(value)
            elif kind == "float":
                if isinstance(value, bool):
                    raise ValueError("must be a number")
                value = float(value)
            elif kind == "choice" and value not in spec.get("choices", []):
                raise ValueError(f"must be one of {spec.get('choices', [])}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid '{name}': {exc}") from exc
        if spec.get("min") is not None and value < spec["min"]:
            raise ValueError(f"'{name}' must be at least {spec['min']}")
        if spec.get("max") is not None and value > spec["max"]:
            raise ValueError(f"'{name}' must be at most {spec['max']}")
        resolved[name] = value
    return resolved

OBS_TYPE_ALIASES = {
    "builtin": "onehot",     # gridworld one-hot states
    "gym-vector": "vector",
}


def env_obs_type(env_dict: Dict[str, Any]) -> str:
    """Classify an environment's observation space from its metadata."""
    tags = set(env_dict.get("tags") or [])
    if "pixels" in tags or env_dict.get("category") == "minatar":
        return "image"
    return env_dict.get("obs_type", "vector")


def check_compat(agent: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    """Return {compatible, level: ok|warning|blocked, blockers[], warnings[], notes[]}.

    `agent` and `env` are the plain dicts produced by list_agents()/list_envs().
    """
    blockers: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []

    if agent.get("status") != "implemented":
        reason = agent.get("status_reason")
        blockers.append(
            f"{agent.get('label')} is {agent.get('status')}, not trainable yet."
            + (f" {reason}" if reason else "")
        )

    if env.get("available") is False:
        blockers.append(env.get("unavailable_reason") or "This environment is not installed.")

    supports = agent.get("supports") or {}
    supported_obs = set(supports.get("obs_types") or ["vector", "onehot"])
    supported_act = set(supports.get("action_types") or ["discrete"])

    obs_type = env_obs_type(env)
    if obs_type not in supported_obs:
        blockers.append(
            f"{agent['label']} cannot process '{obs_type}' observations "
            f"(supports: {', '.join(sorted(supported_obs))})."
        )

    if obs_type == "image" and "image" in supported_obs:
        warnings.append(
            "Pixel observations are flattened into a vector for an MLP; there is no "
            "convolutional encoder, so learning is slower than published CNN results."
        )

    action_type = env.get("action_type", "discrete")
    if action_type not in supported_act:
        blockers.append(
            f"{agent['label']} only supports {'/'.join(sorted(supported_act))} actions; "
            f"this environment is {action_type}."
        )

    if env.get("action_masked") and not supports.get("action_mask", False):
        blockers.append("This environment requires an algorithm that respects action masks.")

    n_agents = int(env.get("n_agents", 1) or 1)
    if n_agents > 1 and not supports.get("multi_agent"):
        if env.get("category") == "pettingzoo":
            notes.append(
                "Two-player game: the learner plays one seat against a random legal "
                "opponent, so the task is a single-agent MDP."
            )
        else:
            blockers.append("This environment requires multi-agent support.")
    if n_agents == 1 and supports.get("multi_agent"):
        notes.append("Designed for multi-agent play but works on single-agent tasks too.")

    # informational ------------------------------------------------------
    if env.get("difficulty") == "hard":
        notes.append("Hard environment: expect long training times and tune carefully.")
    if env.get("category") == "probe":
        notes.append(
            "Diagnostic probe - meant to verify implementations in seconds, not to be a benchmark."
        )
    rec = env.get("recommended_agents") or []
    unsuitable = env.get("unsuitable_agents") or []
    if rec and agent["name"] in rec:
        notes.append(f"Recommended pairing: {agent['label']} performs well here.")
    if agent["name"] in unsuitable:
        warnings.append(
            f"Documented poor fit: {agent['label']} tends to underperform on this environment."
        )
    if env.get("sparse_reward"):
        notes.append("Sparse rewards make credit assignment hard; value-based methods may need many episodes.")

    level = "blocked" if blockers else ("warning" if warnings else "ok")
    return {
        "compatible": not blockers,
        "level": level,
        "blockers": blockers,
        "warnings": warnings,
        "notes": notes,
    }


def compat_matrix(agent_specs: List[Dict], env_specs: List[Dict],
                  include_planned: bool = False) -> Dict[str, Dict[str, Any]]:
    """Full matrix keyed by agent name then env name. Used by /api/capability-matrix."""
    out: Dict[str, Dict[str, Any]] = {}
    for a in agent_specs:
        if not include_planned and a.get("status") != "implemented":
            continue
        row = {}
        for e in env_specs:
            row[e["name"]] = check_compat(a, e)
        out[a["name"]] = {"label": a["label"], "status": a.get("status"), "envs": row}
    return out
