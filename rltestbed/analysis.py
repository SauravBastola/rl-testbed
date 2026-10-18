"""Aggregation of run histories across seeds.

A single RL run is an anecdote; the same configuration over several seeds is
a result. This module turns per-episode histories into per-group curves with
mean, standard deviation and range, plus summary statistics. It is used by
``GET /api/groups/{id}`` and ``GET /api/compare/aggregate`` so the frontend
never re-implements the statistics.
"""

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

GROUP_BY = ("config", "group", "run")


def config_signature(run: Dict[str, Any]) -> str:
    """Runs that differ only by seed (and name/notes) share a signature."""
    params = json.dumps(run.get("params") or {}, sort_keys=True)
    return f"{run['agent']}|{run['env']}|{params}"


def group_key(run: Dict[str, Any], group_by: str) -> str:
    if group_by == "run":
        return run["id"]
    if group_by == "group" and run.get("group_id"):
        return run["group_id"]
    return config_signature(run)


def _series(history: List[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """(episodes, values) for a metric; eval values are sparse."""
    xs, ys = [], []
    for row in history:
        if key == "eval":
            ev = row.get("eval")
            value = ev.get("episode_mean") if isinstance(ev, dict) else None
        else:
            value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            xs.append(int(row.get("episode", len(xs) + 1)))
            ys.append(float(value))
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def _binned(xs: np.ndarray, ys: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ys inside each [edges[i], edges[i+1]) episode bin (NaN if empty)."""
    out = np.full(len(edges) - 1, np.nan)
    if len(xs) == 0:
        return out
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, len(edges) - 2)
    sums = np.bincount(idx, weights=ys, minlength=len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def aggregate_curves(histories: List[List[Dict[str, Any]]], key: str = "reward",
                     max_points: int = 400) -> List[Dict[str, Any]]:
    """Mean / std / min / max across runs, per episode bin.

    Runs of unequal length are allowed: each point reports ``n``, the number of
    runs contributing to it, and std is only defined where n >= 2.
    """
    series = [_series(h, key) for h in histories]
    series = [s for s in series if len(s[0])]
    if not series:
        return []
    last = int(max(s[0].max() for s in series))
    first = int(min(s[0].min() for s in series))
    n_bins = max(1, min(max_points, last - first + 1))
    edges = np.linspace(first, last + 1, n_bins + 1)
    stacked = np.vstack([_binned(xs, ys, edges) for xs, ys in series])
    centers = (edges[:-1] + edges[1:] - 1) / 2.0
    points = []
    for i, x in enumerate(centers):
        col = stacked[:, i]
        col = col[np.isfinite(col)]
        if not len(col):
            continue
        points.append({
            "x": round(float(x), 2),
            "mean": float(col.mean()),
            "std": float(col.std(ddof=1)) if len(col) > 1 else None,
            "min": float(col.min()),
            "max": float(col.max()),
            "n": int(len(col)),
        })
    return points


def _episodes_to_threshold(history: List[Dict[str, Any]], threshold: float) -> Optional[int]:
    for row in history:
        value = row.get("mean_reward")
        if isinstance(value, (int, float)) and value >= threshold:
            return int(row.get("episode", 0))
    return None


def run_summary(run: Dict[str, Any], history: List[Dict[str, Any]],
                threshold: Optional[float] = None) -> Dict[str, Any]:
    rewards = [r["reward"] for r in history if isinstance(r.get("reward"), (int, float))]
    tail = rewards[-max(1, len(rewards) // 10):] if rewards else []
    started, finished = run.get("started_at"), run.get("finished_at")
    return {
        "run_id": run["id"],
        "seed": run.get("seed"),
        "status": run.get("status"),
        "episodes": len(history),
        "final": float(np.mean(tail)) if tail else None,
        "best_eval": run.get("best_eval_reward"),
        "episodes_to_threshold": (
            _episodes_to_threshold(history, threshold) if threshold is not None else None
        ),
        "wall_clock": (finished - started) if started and finished else None,
    }


def _stat(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "n": len(arr)}


def aggregate(items: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
              key: str = "reward", group_by: str = "config",
              threshold: Optional[float] = None, max_points: int = 400) -> List[Dict]:
    """Group (run, history) pairs and aggregate each group."""
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {GROUP_BY}")
    groups: Dict[str, List[Tuple[Dict, List]]] = {}
    for run, hist in items:
        groups.setdefault(group_key(run, group_by), []).append((run, hist))

    out = []
    for gkey, members in groups.items():
        runs = [m[0] for m in members]
        head = runs[0]
        summaries = [run_summary(r, h, threshold) for r, h in members]
        reached = [s["episodes_to_threshold"] for s in summaries]
        out.append({
            "key": gkey,
            "label": head["name"] if len(runs) == 1 else f"{head['agent']} on {head['env']}",
            "agent": head["agent"],
            "env": head["env"],
            "group_id": head.get("group_id"),
            "run_ids": [r["id"] for r in runs],
            "seeds": [r.get("seed") for r in runs],
            "n_runs": len(runs),
            "params": head.get("params") or {},
            "points": aggregate_curves([m[1] for m in members], key, max_points),
            "summary": {
                "final": _stat(s["final"] for s in summaries),
                "best_eval": _stat(s["best_eval"] for s in summaries),
                "wall_clock": _stat(s["wall_clock"] for s in summaries),
                "episodes_to_threshold": _stat(reached),
                "reached_threshold": sum(1 for r in reached if r is not None),
            },
            "runs": summaries,
        })
    return out


def config_diff(runs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Fields whose value differs between runs (params, seed, episodes, agent, env)."""
    fields: Dict[str, List[Any]] = {}
    for key in ("agent", "env", "seed", "episodes"):
        fields[key] = [r.get(key) for r in runs]
    names = sorted({k for r in runs for k in (r.get("params") or {})})
    for name in names:
        fields[f"params.{name}"] = [(r.get("params") or {}).get(name) for r in runs]
    return {k: v for k, v in fields.items() if len({json.dumps(x) for x in v}) > 1}
