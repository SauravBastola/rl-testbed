"""FastAPI application exposing the testbed over REST + WebSocket and
serving the compiled web frontend."""

import asyncio
import json
import os as _os
# Force headless SDL for server-side rgb_array rendering (avoids Cocoa main-thread crash on macOS)
_os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
_os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
_os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from rltestbed.config import (
    RUNS_DIR,
    WEB_DIST_DIR,
    PROJECT_ROOT,
    run_dir as run_dir_for,
)
from rltestbed.core.agents import list_agents, default_params
from rltestbed.core.envs import list_envs, get_env_spec, make_env, discover_custom_envs
from rltestbed.jobs import JOB_MANAGER
from rltestbed.registry import (
    create_run,
    update_run,
    get_run,
    list_runs,
    delete_run,
    keep_model,
    list_models,
    get_model,
    delete_model,
    update_model,
    import_model_file,
)
from rltestbed.benchmarks import benchmark_dict
from rltestbed.capabilities import check_compat, compat_matrix, validate_params
from rltestbed.content import GLOSSARY, LEARNING_PATH
from pydantic import ValidationError

from rltestbed import launch
from rltestbed.schemas import (
    EvaluationRequest,
    ModelUpdateRequest,
    RunCreateRequest,
    RunUpdateRequest,
    SweepCreateRequest,
)
from rltestbed import analysis, labs as labs_mod
from rltestbed.benchmarks import diagnostic_params, verdict as benchmark_verdict
from rltestbed.registry import list_groups, new_group_id
from rltestbed import registry as _registry
from rltestbed import sweeps as sweeps_mod

TERMINAL = ("completed", "stopped", "failed", "interrupted")


def _env_index() -> Dict[str, Dict[str, Any]]:
    return {e["name"]: e for e in list_envs()}


def _with_lab(run: Dict[str, Any], index: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    env = (index or _env_index()).get(run.get("env"))
    run["lab"] = env["lab"] if env else None
    return run


def _progress(run_id: str) -> Dict[str, Any]:
    path = run_dir_for(run_id) / "progress.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _history(run_id: str) -> list:
    path = run_dir_for(run_id) / "history.jsonl"
    out = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    return out


LOOPBACK_HOSTS = ["localhost", "127.0.0.1", "[::1]", "::1"]


def _env_list(name: str) -> Optional[List[str]]:
    raw = _os.environ.get(name, "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else None


def create_app(web_dist: Optional[Path] = None,
               allowed_hosts: Optional[List[str]] = None,
               cors_origins: Optional[List[str]] = None) -> FastAPI:
    """Build the API + web app.

    Security model: a local, single-user laboratory with no authentication.
    - The web app is same-origin and the Vite dev server proxies /api, so no
      cross-origin access is granted by default. (v3.0 allowed every origin,
      which let any website open in the browser call this API and read the
      responses.) Opt in with ``cors_origins`` or RLTESTBED_CORS_ORIGINS.
    - Only requests whose Host header is a loopback name are served, which
      blocks DNS-rebinding attacks. ``rltestbed serve --host 0.0.0.0`` widens
      this deliberately; RLTESTBED_ALLOWED_HOSTS overrides it.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        async def pump_forever():
            while True:
                try:
                    await asyncio.to_thread(JOB_MANAGER.pump)
                except Exception:  # pragma: no cover - keep the loop alive
                    pass
                await asyncio.sleep(1.0)

        task = asyncio.create_task(pump_forever())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="RL Testbed", version="3.1.0", lifespan=lifespan)
    hosts = allowed_hosts or _env_list("RLTESTBED_ALLOWED_HOSTS") or LOOPBACK_HOSTS + ["testserver"]
    if "*" not in hosts:
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    origins = cors_origins or _env_list("RLTESTBED_CORS_ORIGINS")
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    JOB_MANAGER.reconcile()

    # ---------------------------------------------------------- metadata --
    @app.get("/api/envs")
    def api_envs(resolve: bool = False):
        discover_custom_envs()
        return list_envs(resolve=resolve)

    @app.get("/api/envs/{name}")
    def api_env_detail(name: str):
        if name not in get_registry():
            raise HTTPException(404, f"unknown env '{name}'")
        from rltestbed.core.envs import REGISTRY

        return REGISTRY[name]["spec"].to_dict(resolve=True)

    def get_registry():
        from rltestbed.core.envs import REGISTRY

        return REGISTRY

    @app.get("/api/agents")
    def api_agents():
        return list_agents()

    @app.get("/api/agents/compare/code")
    def api_compare_code(ids: str = ""):
        """Compare source code for multiple algorithms side-by-side."""
        from rltestbed.core.agents import AGENT_REGISTRY
        import inspect
        from pathlib import Path as _P
        out = []
        for nid in [x.strip() for x in ids.split(",") if x.strip()]:
            if nid not in AGENT_REGISTRY:
                continue
            cls = AGENT_REGISTRY[nid]
            try:
                source = inspect.getsource(cls)
                file = inspect.getfile(cls)
                lineno = inspect.getsourcelines(cls)[1]
                rel = _P(file).relative_to(PROJECT_ROOT)
                out.append({"name": nid, "file": str(rel), "line": lineno, "source": source, "params": cls.SPEC.get("params", []), "label": cls.SPEC.get("label", nid)})
            except Exception as e:
                out.append({"name": nid, "error": str(e)})
        return out

    @app.get("/api/agents/{name}/code")
    def api_agent_code(name: str):
        """Return actual source code for an algorithm — file_path:line for single source learning."""
        from rltestbed.core.agents import AGENT_REGISTRY
        import inspect
        from pathlib import Path as _P
        if name not in AGENT_REGISTRY:
            raise HTTPException(404, f"no code for '{name}' (planned or unknown)")
        cls = AGENT_REGISTRY[name]
        try:
            source = inspect.getsource(cls)
            file = inspect.getfile(cls)
            lineno = inspect.getsourcelines(cls)[1]
            rel = _P(file).relative_to(PROJECT_ROOT)
            full = _P(file).read_text()[:120000]
            return {"file": str(rel), "line": lineno, "source": source, "full": full}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/api/agents/{name}")
    def api_agent_detail(name: str):
        spec = next((item for item in list_agents() if item["name"] == name), None)
        if not spec:
            raise HTTPException(404, f"unknown algorithm '{name}'")
        return spec

    @app.get("/api/capabilities")
    def api_capabilities(include_planned: bool = True):
        agents = list_agents(include_planned=include_planned)
        envs = list_envs()
        return {"agents": agents, "environments": envs,
                "matrix": compat_matrix(agents, envs, include_planned=include_planned)}

    @app.get("/api/compatibility")
    def api_compatibility(agent: str, env: str):
        agent_spec = next((item for item in list_agents() if item["name"] == agent), None)
        env_spec = next((item for item in list_envs() if item["name"] == env), None)
        if not agent_spec or not env_spec:
            raise HTTPException(404, "unknown algorithm or environment")
        return check_compat(agent_spec, env_spec)

    @app.get("/api/benchmarks")
    def api_benchmarks():
        runs = list_runs(limit=500)
        return [
            {**benchmark, "history": [
                {"id": run["id"], "agent": run["agent"], "status": run["status"],
                 "best_eval_reward": run.get("best_eval_reward"),
                 "created_at": run["created_at"]}
                for run in runs if run["env"] == benchmark["probe"]
            ][:10]}
            for benchmark in benchmark_dict()
        ]

    @app.get("/api/learning-path")
    def api_learning_path():
        return LEARNING_PATH

    @app.get("/api/curriculum")
    def api_curriculum():
        """Learning-path modules with each step's live status from the user's runs."""
        from rltestbed import curriculum

        return curriculum.with_status(list_runs(limit=5000), _env_index())

    @app.get("/api/glossary")
    def api_glossary():
        return GLOSSARY

    @app.get("/api/health")
    def api_health():
        return {"status": "ok", "database": "ready"}

    @app.get("/api/labs")
    def api_labs():
        return labs_mod.labs_with_members(list_envs(), list_agents())

    @app.get("/api/system")
    def api_system():
        import platform
        import shutil as _sh
        import sys as _sys

        from rltestbed.config import DB_PATH, MODELS_DIR
        from rltestbed.core.envs.availability import installed_extras

        def _size(path: Path) -> int:
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0

        try:
            import torch

            torch_info = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
        except Exception as exc:  # pragma: no cover
            torch_info = {"version": None, "error": str(exc)}
        disk = _sh.disk_usage(str(PROJECT_ROOT))
        envs = list_envs()
        return {
            "version": app.version,
            "python": _sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch_info,
            "paths": {"root": str(PROJECT_ROOT), "database": str(DB_PATH),
                      "runs": str(RUNS_DIR), "models": str(MODELS_DIR)},
            "usage": {"runs_bytes": _size(RUNS_DIR), "models_bytes": _size(MODELS_DIR),
                      "disk_free_bytes": disk.free},
            "extras": installed_extras(),
            "jobs": {"running": JOB_MANAGER.running_count(),
                     "max_concurrent": JOB_MANAGER.max_concurrent,
                     "queued": len(list_runs(status="queued"))},
            "catalogue": {
                "envs": len(envs),
                "envs_available": sum(1 for e in envs if e.get("available")),
                "agents": len(list_agents()),
                "agents_implemented": sum(1 for a in list_agents() if a["status"] == "implemented"),
            },
        }

    # ----------------------------------------------------------- groups --
    def _items(runs):
        return [(r, _history(r["id"])) for r in runs]

    @app.get("/api/groups")
    def api_groups():
        return list_groups()

    @app.get("/api/groups/{group_id}")
    def api_group(group_id: str, metric: str = "reward", max_points: int = 400):
        runs = list_runs(group_id=group_id)
        if not runs:
            raise HTTPException(404, f"unknown group '{group_id}'")
        env = _env_index().get(runs[0]["env"]) or {}
        groups = analysis.aggregate(_items(runs), key=metric, group_by="group",
                                    threshold=env.get("reward_threshold"),
                                    max_points=max_points)
        return {"group_id": group_id, "runs": runs, "metric": metric,
                "threshold": env.get("reward_threshold"), **groups[0]}

    @app.get("/api/compare/aggregate")
    def api_compare_aggregate(run_ids: str, metric: str = "reward",
                              group_by: str = "config", max_points: int = 400,
                              format: str = "json"):
        """Overlay runs, grouping seeds of the same configuration into mean ± std."""
        runs = [r for r in (get_run(x.strip()) for x in run_ids.split(",") if x.strip()) if r]
        if not runs:
            return {"groups": [], "diff": {}, "metric": metric, "threshold": None}
        if group_by not in analysis.GROUP_BY:
            raise HTTPException(422, f"group_by must be one of {analysis.GROUP_BY}")
        index = _env_index()
        thresholds = {index.get(r["env"], {}).get("reward_threshold") for r in runs}
        threshold = thresholds.pop() if len(thresholds) == 1 else None
        metric_threshold = threshold if metric in ("reward", "mean_reward", "eval") else None
        groups = analysis.aggregate(_items(runs), key=metric, group_by=group_by,
                                    threshold=threshold, max_points=max_points)
        if format == "csv":
            import csv
            import io

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["group", "n_runs", "episode", "mean", "std", "min", "max", "n"])
            for g in groups:
                for p in g["points"]:
                    writer.writerow([g["label"], g["n_runs"], p["x"], p["mean"],
                                     "" if p["std"] is None else p["std"], p["min"], p["max"], p["n"]])
            return _attachment(buf.getvalue(), f"compare-{metric}.csv", "text/csv")
        return {
            "metric": metric,
            "group_by": group_by,
            "threshold": metric_threshold,
            "groups": groups,
            "diff": analysis.config_diff(runs),
        }

    # ----------------------------------------------------------- sweeps --
    def _sweep_body(body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return SweepCreateRequest(**body).model_dump()
        except ValidationError as exc:
            reasons = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
            raise HTTPException(422, {"message": "invalid sweep", "reasons": reasons}) from exc

    def _sweep_request(body: Dict[str, Any]):
        try:
            return launch.check_sweep(_sweep_body(body))
        except launch.LaunchError as exc:
            raise HTTPException(exc.status, exc.detail()) from exc

    @app.post("/api/sweeps/preview")
    def api_sweep_preview(body: Dict[str, Any]):
        """Expand a sweep without creating anything; errors come back as data."""
        try:
            _, _, configs, seeds, _ = _sweep_request(body)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"reasons": [str(exc.detail)]}
            return {"valid": False, "errors": detail.get("reasons", []), "n_configs": 0, "n_runs": 0}
        return {"valid": True, "errors": [], "n_configs": len(configs),
                "n_runs": len(configs) * len(seeds), "configs": configs[:50],
                "max_runs": sweeps_mod.MAX_RUNS}

    @app.post("/api/sweeps")
    def api_create_sweep(body: Dict[str, Any]):
        try:
            return launch.create_sweep_runs(_sweep_body(body))
        except launch.LaunchError as exc:
            raise HTTPException(exc.status, exc.detail()) from exc

    @app.get("/api/sweeps")
    def api_sweeps(lab: Optional[str] = None):
        envs = None
        if lab:
            envs = [n for n, e in _env_index().items() if e["lab"] == lab]
        out = _registry.list_sweeps(envs=envs)
        for sw in out:
            sw["swept"] = sorted((sw["request"].get("space") or {}).keys())
        return out

    @app.get("/api/sweeps/{sweep_id}")
    def api_sweep(sweep_id: str, rank_by: str = "final"):
        sw = _registry.get_sweep(sweep_id)
        if not sw:
            raise HTTPException(404, f"unknown sweep '{sweep_id}'")
        if rank_by not in ("final", "best_eval"):
            raise HTTPException(422, "rank_by must be 'final' or 'best_eval'")
        runs = list_runs(sweep_id=sweep_id, limit=5000)
        env = _env_index().get(sw["env"]) or {}
        groups = analysis.aggregate([(r, _history(r["id"])) for r in runs], key="reward",
                                    group_by="group", threshold=env.get("reward_threshold"),
                                    max_points=150)
        swept = sw["request"].get("space") or {}
        by_group = {r["group_id"]: r for r in runs}
        status_of = {r["id"]: r["status"] for r in runs}
        configs = []
        for g in groups:
            params = by_group[g["group_id"]]["params"] or {}
            statuses = [status_of[x] for x in g["run_ids"]]
            configs.append({
                "group_id": g["group_id"],
                "params": {k: params.get(k) for k in swept},
                "all_params": params,
                "run_ids": g["run_ids"],
                "seeds": g["seeds"],
                "n_runs": g["n_runs"],
                "done": sum(st in TERMINAL for st in statuses),
                "running": sum(st in ("running", "stopping") for st in statuses),
                "summary": g["summary"],
                "score": g["summary"][rank_by],
                "points": g["points"],
            })
        configs.sort(key=lambda c: (c["score"]["mean"] is None, -(c["score"]["mean"] or 0.0)))
        for i, c in enumerate(configs):
            c["rank"] = i + 1
        # stable index = creation order, for the effects scatter
        order = sorted(configs, key=lambda c: min(c["run_ids"]))
        for i, c in enumerate(order):
            c["index"] = i + 1
        counts = {k: sum(r["status"] in v for r in runs) for k, v in {
            "completed": ("completed",), "running": ("running", "stopping"),
            "queued": ("queued", "created"), "failed": ("failed", "interrupted"),
            "stopped": ("stopped",)}.items()}
        return {**sw, "rank_by": rank_by, "threshold": env.get("reward_threshold"),
                "n_runs": len(runs), "counts": counts, "lab": env.get("lab"),
                "configs": configs, "effects": sweeps_mod.effects(configs, swept)}

    @app.post("/api/sweeps/{sweep_id}/stop")
    def api_stop_sweep(sweep_id: str):
        if not _registry.get_sweep(sweep_id):
            raise HTTPException(404)
        import time as _t

        stopped = 0
        for r in list_runs(sweep_id=sweep_id, limit=5000):
            if r["status"] in ("queued", "created"):
                update_run(r["id"], status="stopped", finished_at=_t.time())
                stopped += 1
            elif r["status"] == "running" and JOB_MANAGER.stop(r["id"]):
                stopped += 1
        return {"stopped": stopped}

    @app.delete("/api/sweeps/{sweep_id}")
    async def api_delete_sweep(sweep_id: str):
        if not _registry.get_sweep(sweep_id):
            raise HTTPException(404)
        runs = list_runs(sweep_id=sweep_id, limit=5000)
        for r in runs:
            await api_delete_run(r["id"])
        _registry.delete_sweep(sweep_id)
        return {"deleted": sweep_id, "runs_deleted": len(runs)}

    # ------------------------------------------------------ experiments --
    from rltestbed import experiments as experiments_mod

    def _attachment(text: str, filename: str, media_type: str) -> Response:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in filename)[:120]
        return Response(text, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="{safe}"'})

    @app.post("/api/experiments/validate")
    def api_experiment_validate(body: Dict[str, Any]):
        """Validate an experiment file. Errors come back as data, with locations."""
        result = experiments_mod.parse(str(body.get("text") or ""))
        return {k: v for k, v in result.items() if k != "experiment"}

    @app.post("/api/experiments/launch")
    def api_experiment_launch(body: Dict[str, Any]):
        try:
            return experiments_mod.launch_file(str(body.get("text") or ""))
        except launch.LaunchError as exc:
            raise HTTPException(exc.status, exc.detail()) from exc

    @app.get("/api/experiments/examples")
    def api_experiment_examples():
        return experiments_mod.examples(labs_mod.labs_with_members(list_envs(), list_agents()))

    @app.get("/api/experiments/export")
    def api_experiment_export(run_ids: str = "", sweep_id: str = "", name: str = ""):
        """An experiment file that recreates the given runs and/or sweep."""
        runs = [r for r in (get_run(x.strip()) for x in run_ids.split(",") if x.strip()) if r]
        sweeps = []
        if sweep_id:
            sw = _registry.get_sweep(sweep_id)
            if not sw:
                raise HTTPException(404, f"unknown sweep '{sweep_id}'")
            sweeps.append(sw)
        if not runs and not sweeps:
            raise HTTPException(404, "nothing to export")
        title = name or experiments_mod.default_name(runs, sweeps)
        filename = f"{title}.yaml".replace(" ", "-")
        text = experiments_mod.dump(title, runs=runs, sweeps=sweeps, filename=filename)
        return _attachment(text, filename, "application/x-yaml")

    @app.get("/api/runs/{run_id}/history.csv")
    def api_run_history_csv(run_id: str):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        rows = _history(run_id)
        import csv
        import io

        metric_cols = sorted({k for r in rows for k in r if k.startswith("metric_")})
        cols = ["episode", "reward", "mean_reward", "steps", "eval_mean", "eval_std"] + metric_cols
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(cols)
        for r in rows:
            ev = r.get("eval") if isinstance(r.get("eval"), dict) else {}
            values = {**r, "eval_mean": ev.get("episode_mean"), "eval_std": ev.get("episode_std")}
            writer.writerow(["" if values.get(c) is None else values.get(c) for c in cols])
        return _attachment(buf.getvalue(), f"{run['name']}-{run_id}.csv", "text/csv")

    # ------------------------------------------------------ diagnostics --
    @app.get("/api/diagnostics")
    def api_diagnostics():
        """Benchmark x algorithm verdict matrix from the latest run of each pair."""
        agents = [a for a in list_agents(include_planned=False) if a["name"] != "random"]
        envs = _env_index()
        runs = list_runs(limit=2000, envs=[b["probe"] for b in benchmark_dict()])
        latest: Dict[tuple, Dict] = {}
        for r in runs:  # newest first
            latest.setdefault((r["env"], r["agent"]), r)
        rows = []
        for b in benchmark_dict():
            env = envs.get(b["probe"], {})
            cells = {}
            for a in agents:
                compat = check_compat(a, env) if env else {"compatible": False}
                run = latest.get((b["probe"], a["name"]))
                cells[a["name"]] = {
                    "compatible": compat["compatible"],
                    "expected": a["name"] in b["should_pass"],
                    "verdict": benchmark_verdict(b, run) if compat["compatible"] else "n/a",
                    "run_id": run["id"] if run else None,
                    "best_eval": run.get("best_eval_reward") if run else None,
                    "finished_at": run.get("finished_at") if run else None,
                }
            rows.append({**b, "cells": cells})
        return {"agents": [{"name": a["name"], "label": a["label"], "category": a["category"]}
                           for a in agents], "benchmarks": rows}

    @app.post("/api/diagnostics/run")
    def api_diagnostics_run(body: Dict[str, Any] | None = None):
        """Queue probe runs. Defaults: every expected (benchmark, agent) pair not yet passing."""
        body = body or {}
        want_agents = set(body.get("agents") or [])
        want_bench = set(body.get("benchmarks") or [])
        only_missing = body.get("only_missing", True)
        matrix = api_diagnostics()
        agent_specs = {a["name"]: a for a in list_agents(include_planned=False)}
        created = []
        for row in matrix["benchmarks"]:
            if want_bench and row["id"] not in want_bench:
                continue
            for name, cell in row["cells"].items():
                if want_agents and name not in want_agents:
                    continue
                if not want_agents and not cell["expected"]:
                    continue
                if not cell["compatible"] or cell["verdict"] == "running":
                    continue
                if only_missing and cell["verdict"] == "pass":
                    continue
                params = validate_params(agent_specs[name], diagnostic_params(agent_specs[name]))
                rid = create_run(f"{name}-{row['probe']}", name, row["probe"], 0,
                                 row["episodes"], params,
                                 {"episodes": row["episodes"], "seed": 0, "eval_freq": 10,
                                  "eval_episodes": 10, "checkpoint_freq": 1000,
                                  "save_best": False, "save_final": False},
                                 tags=["diagnostic"])
                JOB_MANAGER.enqueue(rid)
                created.append(rid)
        return {"run_ids": created, "queued": len(created)}

    # -------------------------------------------------------------- runs --
    @app.post("/api/runs")
    def api_create_run(body: RunCreateRequest):
        try:
            # `run_id` is the original contract; `run_ids` / `group_id` are additive.
            return launch.create_runs(body.model_dump())
        except launch.LaunchError as exc:
            raise HTTPException(exc.status, exc.detail()) from exc

    @app.get("/api/runs")
    def api_list_runs(lab: Optional[str] = None, group_id: Optional[str] = None,
                      status: Optional[str] = None, env: Optional[str] = None,
                      agent: Optional[str] = None, sweep_id: Optional[str] = None,
                      limit: int = 500):
        index = _env_index()
        envs = None
        if lab:
            envs = [name for name, e in index.items() if e["lab"] == lab]
        runs = list_runs(limit=limit, group_id=group_id, status=status, env=env,
                         agent=agent, envs=envs, sweep_id=sweep_id)
        for r in runs:
            _with_lab(r, index)
            if r["status"] in ("running", "stopping"):
                r["progress"] = _progress(r["id"])
        return runs

    @app.get("/api/runs/{run_id}")
    def api_get_run(run_id: str):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        run["progress"] = _progress(run_id)
        run["live"] = JOB_MANAGER.is_running(run_id)
        return _with_lab(run)

    @app.patch("/api/runs/{run_id}")
    def api_update_run(run_id: str, body: RunUpdateRequest):
        if not get_run(run_id):
            raise HTTPException(404)
        fields = body.model_dump(exclude_none=True)
        if fields:
            update_run(run_id, **fields)
        return _with_lab(get_run(run_id))

    @app.get("/api/runs/{run_id}/history")
    def api_run_history(run_id: str, since: int = 0):
        """Per-episode history. ``since`` returns only episodes after that number,
        which lets a live view poll cheaply."""
        if not get_run(run_id):
            raise HTTPException(404)
        hist = _history(run_id)
        if since > 0:
            hist = [h for h in hist if int(h.get("episode", 0)) > since]
        return hist

    @app.post("/api/runs/{run_id}/start")
    def api_start_run(run_id: str):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        if run["status"] not in ("created", "stopped", "interrupted", "failed"):
            raise HTTPException(409, f"cannot start a run in status '{run['status']}'")
        # history.jsonl is appended to, so clear the previous attempt's files
        import shutil

        shutil.rmtree(run_dir_for(run_id), ignore_errors=True)
        update_run(run_id, status="created", finished_at=None, pid=None)
        state = JOB_MANAGER.enqueue(run_id)
        return {"status": "started" if state == "running" else "queued"}

    @app.post("/api/runs/{run_id}/stop")
    def api_stop_run(run_id: str):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        if run["status"] in ("queued", "created"):
            import time as _t

            update_run(run_id, status="stopped", finished_at=_t.time())
            return {"status": "stopped"}
        ok = JOB_MANAGER.stop(run_id)
        return {"status": "stopping" if ok else "not running"}

    @app.delete("/api/runs/{run_id}")
    async def api_delete_run(run_id: str):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        if JOB_MANAGER.is_running(run_id):
            JOB_MANAGER.stop(run_id)
            await asyncio.sleep(0.5)
        import shutil

        shutil.rmtree(run_dir_for(run_id), ignore_errors=True)
        delete_run(run_id)
        return {"deleted": run_id}

    @app.post("/api/runs/{run_id}/keep-model")
    def api_keep_model(run_id: str, body: Dict[str, Any] | None = None):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        source = (body or {}).get("source", "best")
        path = run_dir_for(run_id) / (
            "best.pth" if source == "best" else "checkpoint.pth"
        )
        if not path.exists():
            raise HTTPException(
                409,
                f"no {'best' if source == 'best' else 'latest'} checkpoint saved yet",
            )
        model = keep_model(
            run_id,
            path,
            name=(body or {}).get("name"),
            description=(body or {}).get("description", ""),
        )
        return model

    @app.get("/api/compare")
    def api_compare(run_ids: str = "", model_ids: str = ""):
        """Return histories for multiple runs/models for comparison workspace."""
        result: Dict[str, Any] = {"runs": [], "models": []}
        if run_ids:
            for rid in [x.strip() for x in run_ids.split(",") if x.strip()]:
                run = get_run(rid)
                if not run:
                    continue
                hist = _history(rid)
                result["runs"].append({"run": run, "history": hist})
        if model_ids:
            for mid in [x.strip() for x in model_ids.split(",") if x.strip()]:
                m = get_model(mid)
                if m:
                    result["models"].append(m)
        return result

    @app.post("/api/runs/{run_id}/clone")
    def api_clone_run(run_id: str):
        run = get_run(run_id)
        if not run:
            raise HTTPException(404)
        config = run.get("config", {}) or {}
        # Recreate run with same params but new id
        from rltestbed.registry import create_run as _create

        new_id = _create(
            name=f"{run['name']}-clone",
            agent=run["agent"],
            env=run["env"],
            seed=run["seed"],
            episodes=run["episodes"],
            params=run.get("params", {}),
            config=config,
            tags=run.get("tags", []),
            notes=run.get("notes", ""),
            parent_run_id=run_id,
            device=run.get("device", "cpu"),
        )
        JOB_MANAGER.enqueue(new_id)
        return {"run_id": new_id, "parent": run_id}

    @app.get("/api/envs/{name}/preview")
    def api_env_preview(name: str, episodes: int = 1, steps: int = 20):
        """Run random policy for a few steps and return trajectory for interactive preview."""
        if name not in get_registry():
            raise HTTPException(404, f"unknown env '{name}'")
        import numpy as np

        env = make_env(name)
        trajectory = []
        for ep in range(min(episodes, 3)):
            obs = env.reset()
            ep_steps = []
            done = False
            s = 0
            while not done and s < min(steps, 50):
                # capture observation as list
                if isinstance(obs, dict):
                    obs_repr = {"obs": list(np.asarray(obs.get("obs", [])).flatten()[:20].tolist()), "action_mask": obs.get("action_mask")}
                else:
                    try:
                        obs_repr = list(np.asarray(obs, dtype=float).flatten()[:20].tolist())
                    except Exception:
                        obs_repr = str(obs)[:200]
                action = int(np.random.randint(0, env.act_dim))
                # respect masks if present
                if isinstance(obs, dict) and obs.get("action_mask"):
                    valid = [i for i, v in enumerate(obs["action_mask"]) if v]
                    if valid:
                        action = int(np.random.choice(valid))
                next_obs, reward, term, trunc, info = env.step(action)
                ep_steps.append({"obs": obs_repr, "action": action, "reward": float(reward), "terminated": bool(term), "truncated": bool(trunc)})
                obs = next_obs
                done = term or trunc
                s += 1
            trajectory.append({"episode": ep, "steps": ep_steps, "length": len(ep_steps)})
        env.close()
        # include spec for UI
        spec = get_registry()[name]["spec"].to_dict(resolve=False)
        return {"env": spec["name"], "label": spec["label"], "trajectory": trajectory}

    @app.post("/api/rollout")
    def api_rollout(body: Dict[str, Any]):
        """Play episodes for the Visualize page (see rltestbed/rollout.py).

        MuJoCo renders in a subprocess, whose main thread can own an OpenGL
        context on every platform; everything else runs in-process.
        """
        from rltestbed import rollout

        env_name = body.get("env")
        if body.get("model_id"):
            model = get_model(body["model_id"])
            env_name = model["env"] if model else env_name
        try:
            if env_name and rollout.needs_isolation(env_name):
                return rollout.run_isolated(body)
            return rollout.run(body)
        except rollout.RolloutError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @app.get("/api/runs/{run_id}/logs")
    def api_run_logs(run_id: str, tail: int = 200):
        if not get_run(run_id):
            raise HTTPException(404)
        path = run_dir_for(run_id) / "worker.log"
        if not path.exists():
            return {"lines": [], "exists": False}
        try:
            lines = path.read_text().splitlines()[-tail:]
        except Exception:
            lines = []
        return {"lines": lines, "exists": True}

    @app.get("/api/runs/{run_id}/artifacts")
    def api_run_artifacts(run_id: str):
        if not get_run(run_id):
            raise HTTPException(404)
        d = run_dir_for(run_id)
        if not d.exists():
            return {"files": []}
        files = []
        for p in sorted(d.rglob("*")):
            if p.is_file():
                try:
                    files.append({"path": str(p.relative_to(d)), "size": p.stat().st_size})
                except Exception:
                    pass
        return {"files": files}

    @app.get("/api/runs/{run_id}/artifacts/{path:path}")
    def api_run_artifact(run_id: str, path: str):
        if not get_run(run_id):
            raise HTTPException(404)
        base = run_dir_for(run_id).resolve()
        target = (base / path).resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(404, "no such artifact")
        return FileResponse(target, filename=target.name)

    # ------------------------------------------------------------ models --
    @app.get("/api/models")
    def api_models():
        return list_models()

    @app.post("/api/models/import")
    async def api_import_model(file: UploadFile, agent: str = "dqn", env: str = "cartpole"):
        data = await file.read()
        if len(data) == 0:
            raise HTTPException(422, "empty file")
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(422, "file too large")
        rec = import_model_file(data, agent=agent, env=env, name=file.filename)
        return rec

    @app.patch("/api/models/{model_id}")
    def api_update_model(model_id: str, body: ModelUpdateRequest):
        if not get_model(model_id):
            raise HTTPException(404)
        fields = body.model_dump(exclude_none=True)
        update_model(model_id, **fields)
        return get_model(model_id)

    @app.delete("/api/models/{model_id}")
    def api_delete_model(model_id: str):
        if not get_model(model_id):
            raise HTTPException(404)
        delete_model(model_id)
        return {"deleted": model_id}

    @app.post("/api/models/{model_id}/evaluate")
    def api_evaluate_model(model_id: str, body: EvaluationRequest | None = None):
        import numpy as np

        m = get_model(model_id)
        if not m:
            raise HTTPException(404)
        episodes = body.episodes if body else 20
        from rltestbed.core.agents import create_agent
        from rltestbed.core.envs import REGISTRY

        spec = REGISTRY[m["env"]]["spec"]
        if spec.category == "language":
            # a language model is evaluated by sampling answers and verifying them
            from rltestbed.rollout import run_language

            out = run_language(m["env"], spec, m, n=max(1, episodes) * 10, seed=0)
            rewards = [x["reward"] for x in out["samples"]]
            return {"model_id": model_id, "env": m["env"], "episodes": len(rewards),
                    "mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
                    "min": float(np.min(rewards)), "max": float(np.max(rewards))}

        env = make_env(m["env"])
        agent = create_agent(m["agent"], env.obs_dim, env.act_dim)
        agent.TRAIN = False
        agent.load_model(m["path"])
        rewards = []
        for _ in range(episodes):
            obs = env.reset()
            total, done, steps = 0.0, False, 0
            while not done and steps < 10000:
                a = agent.get_action(obs)
                obs, r, term, trunc, _ = env.step(a)
                total += float(r)
                done = term or trunc
                steps += 1
            rewards.append(total)
        env.close()
        return {
            "model_id": model_id,
            "env": m["env"],
            "episodes": episodes,
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "rewards": [float(x) for x in rewards],
        }

    # ----------------------------------------------------------- websocket --
    @app.websocket("/ws/runs/{run_id}")
    async def ws_run(ws: WebSocket, run_id: str):
        await ws.accept()
        last_payload = None
        try:
            while True:
                run = get_run(run_id)
                if not run:
                    await ws.send_json({"error": "unknown run"})
                    break
                prog = _progress(run_id)
                payload = json.dumps(
                    {
                        "status": run["status"],
                        "live": JOB_MANAGER.is_running(run_id),
                        **prog,
                    },
                    sort_keys=True,
                )
                if payload != last_payload:
                    await ws.send_text(payload)
                    last_payload = payload
                if run["status"] in TERMINAL:
                    # send one final update then linger briefly for late subscribers
                    await asyncio.sleep(2.0)
                    break
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------- static --
    dist = Path(web_dist or WEB_DIST_DIR).resolve()
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            # Serve real files only from inside dist. `full_path` arrives
            # percent-decoded, so "..%2F" becomes "../": resolve and confine it.
            # (v3.0 joined it unchecked and would serve any readable file.)
            file = (dist / full_path).resolve()
            if file.is_file() and dist in file.parents:
                return FileResponse(file)
            if full_path.startswith("api/"):
                raise HTTPException(404, "unknown API endpoint")
            return FileResponse(dist / "index.html")

    return app


app = create_app()
