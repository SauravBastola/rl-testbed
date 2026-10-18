"""Hyperparameter sweeps: expand a search space into run configurations, and
summarise the results.

A sweep is one (algorithm, environment) pair, a search space over that
algorithm's parameters, and a list of seeds. Every configuration becomes a seed
group (one run per seed, sharing a ``group_id``), so results pool into mean ± std
with the same code as everywhere else (:mod:`rltestbed.analysis`).

Search space format (``space``), keyed by parameter name::

    {"lr": {"values": [1e-3, 3e-4]}}               # an explicit list (grid or random)
    {"lr": {"min": 1e-5, "max": 1e-2, "log": true}} # a range (random search only)

Grid search takes the Cartesian product of every ``values`` list. Random search
draws ``n_trials`` configurations: ranges are sampled uniformly (log-uniformly
with ``log``), lists by uniform choice. Every configuration is validated with
:func:`rltestbed.capabilities.validate_params`, the same check as a single run.
"""

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rltestbed.capabilities import validate_params

MAX_RUNS = 200  # configurations × seeds; keeps a sweep within what a laptop can queue
STRATEGIES = ("grid", "random")


class SweepError(ValueError):
    """The search space is invalid; ``errors`` lists every problem found."""

    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _cast(spec: Dict[str, Any], value: Any) -> Any:
    kind = spec["type"]
    if kind == "int":
        return int(round(float(value)))
    if kind == "float":
        return float(value)
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return value


def _check_space(agent: Dict[str, Any], space: Dict[str, Any], strategy: str) -> List[str]:
    params = {p["name"]: p for p in agent.get("params", [])}
    errors = []
    if not space:
        errors.append("Choose at least one parameter to sweep.")
    for name, dim in space.items():
        p = params.get(name)
        if p is None:
            errors.append(f"{agent['label']} has no parameter '{name}'.")
            continue
        if "values" in dim:
            if not isinstance(dim["values"], list) or not dim["values"]:
                errors.append(f"{name}: give at least one value.")
        elif "min" in dim and "max" in dim:
            if strategy == "grid":
                errors.append(f"{name}: grid search needs a list of values, not a range.")
            elif p["type"] not in ("int", "float"):
                errors.append(f"{name}: ranges only apply to numeric parameters.")
            elif float(dim["min"]) >= float(dim["max"]):
                errors.append(f"{name}: min must be below max.")
            elif dim.get("log") and float(dim["min"]) <= 0:
                errors.append(f"{name}: a log range needs min > 0.")
        else:
            errors.append(f"{name}: give either 'values' or 'min' and 'max'.")
    return errors


def expand(agent: Dict[str, Any], space: Dict[str, Any], strategy: str = "grid",
           n_trials: int = 10, base: Optional[Dict[str, Any]] = None,
           seed: int = 0) -> List[Dict[str, Any]]:
    """Return the list of swept-parameter dicts (one per configuration).

    Raises :class:`SweepError` describing every problem when the space is invalid.
    """
    if strategy not in STRATEGIES:
        raise SweepError([f"strategy must be one of {STRATEGIES}"])
    errors = _check_space(agent, space, strategy)
    if errors:
        raise SweepError(errors)
    params = {p["name"]: p for p in agent["params"]}
    names = sorted(space)

    if strategy == "grid":
        size = math.prod(len(space[n]["values"]) for n in names)
        if size > MAX_RUNS:
            raise SweepError([f"The grid has {size} configurations; the limit is {MAX_RUNS}."])
        combos = itertools.product(*[space[n]["values"] for n in names])
        configs = [{n: _cast(params[n], v) for n, v in zip(names, combo)} for combo in combos]
    else:
        if not 1 <= int(n_trials) <= MAX_RUNS:
            raise SweepError([f"Random search needs between 1 and {MAX_RUNS} trials."])
        rng = np.random.default_rng(seed)
        configs = []
        for _ in range(int(n_trials)):
            cfg = {}
            for n in names:
                dim = space[n]
                if "values" in dim:
                    v = dim["values"][int(rng.integers(len(dim["values"])))]
                elif dim.get("log"):
                    v = math.exp(rng.uniform(math.log(float(dim["min"])), math.log(float(dim["max"]))))
                else:
                    v = rng.uniform(float(dim["min"]), float(dim["max"]))
                cfg[n] = _cast(params[n], v)
            configs.append(cfg)

    # de-duplicate (random search over small lists repeats itself), keep order
    seen, unique = set(), []
    for cfg in configs:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)

    problems = []
    for cfg in unique:
        try:
            validate_params(agent, {**(base or {}), **cfg})
        except ValueError as exc:
            problems.append(f"{cfg}: {exc}")
    if problems:
        raise SweepError(problems[:10])
    return unique


def plan(agent: Dict[str, Any], body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Expand a sweep request; returns (configs, seeds) or raises SweepError."""
    seeds = [int(s) for s in body.get("seeds") or [0]]
    if len(set(seeds)) != len(seeds):
        raise SweepError(["Seeds must be unique."])
    configs = expand(agent, body.get("space") or {}, body.get("strategy", "grid"),
                     int(body.get("n_trials") or 10), body.get("base_params") or {},
                     int(body.get("sample_seed") or 0))
    total = len(configs) * len(seeds)
    if total > MAX_RUNS:
        raise SweepError([f"{len(configs)} configurations × {len(seeds)} seeds = {total} runs; "
                          f"the limit is {MAX_RUNS}."])
    return configs, seeds


# ----------------------------------------------------------------- results
def _stat(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "n": len(arr)}


def effects(configs: List[Dict[str, Any]], swept: Dict[str, Any]) -> Dict[str, Any]:
    """Per swept parameter: how the score depends on its value.

    Lists of values become ``discrete`` (mean score per value); random ranges
    become ``continuous`` (one point per configuration, for a scatter plot).
    """
    out = {}
    for name, dim in swept.items():
        scored = [c for c in configs if c["score"]["mean"] is not None]
        if "values" in dim:
            rows = []
            for value in dim["values"]:
                hits = [c["score"]["mean"] for c in scored if c["params"].get(name) == value
                        or (isinstance(value, (int, float)) and isinstance(c["params"].get(name), (int, float))
                            and math.isclose(float(c["params"][name]), float(value), rel_tol=1e-9))]
                rows.append({"value": value, **_stat(hits)})
            out[name] = {"kind": "discrete", "values": rows}
        else:
            out[name] = {
                "kind": "continuous",
                "log": bool(dim.get("log")),
                "points": [{"value": c["params"][name], "score": c["score"]["mean"],
                            "config": c["index"]} for c in scored],
            }
    return out
