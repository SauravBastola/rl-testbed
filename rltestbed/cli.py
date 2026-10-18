"""RL Testbed command line interface.

Usage highlights:
    rltestbed serve                          # start the web app + API
    rltestbed train dqn cartpole             # quick foreground training run
    rltestbed envs                           # list available environments
    rltestbed agents                         # list available agents
    rltestbed runs                           # list runs
    rltestbed runs-show <id>                 # inspect a run
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="RL Testbed - train, track and compare RL agents.",
    no_args_is_help=True,
)

runs_app = typer.Typer(help="Inspect / manage runs", no_args_is_help=True)
models_app = typer.Typer(help="Manage kept models", no_args_is_help=True)
envs_app = typer.Typer(help="Explore environments", no_args_is_help=True)


def _fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"


# ------------------------------------------------------------------ serve --
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="API/web port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (development)"),
):
    """Start the RL Testbed web app + REST API."""
    import uvicorn
    from rltestbed.server.app import create_app

    loopback = host in ("127.0.0.1", "localhost", "::1")
    if loopback:
        app_instance = create_app()
    else:
        # Binding beyond loopback is a deliberate choice to share on a network,
        # so accept any Host header unless RLTESTBED_ALLOWED_HOSTS narrows it.
        import os

        hosts = [h.strip() for h in os.environ.get("RLTESTBED_ALLOWED_HOSTS", "*").split(",")]
        app_instance = create_app(allowed_hosts=hosts)
        typer.secho(
            f"Warning: serving on {host}, reachable from other machines. "
            "RL Testbed has no authentication: anyone who can reach this port can "
            "start, stop and delete runs.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"RL Testbed serving on http://{host}:{port}")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


# ------------------------------------------------------------------ train --
@app.command()
def train(
    agent: str = typer.Argument(..., help=f"Agent name (see 'rltestbed agents')"),
    env: str = typer.Argument(..., help="Environment name (see 'rltestbed envs')"),
    episodes: int = typer.Option(500, help="Number of training episodes"),
    seed: int = typer.Option(42, help="Random seed"),
    name: Optional[str] = typer.Option(None, help="Friendly name for this run"),
    eval_freq: int = typer.Option(25, help="Episodes between evaluations"),
    max_steps: Optional[int] = typer.Option(None, help="Override per-episode step cap"),
    param: List[str] = typer.Option(
        [], "--param", "-p", help="Hyperparameter overrides, e.g. -p lr=0.001"
    ),
    keep: bool = typer.Option(
        False, "--keep", help="Automatically keep (promote) the best model"
    ),
):
    """Train an agent on an environment in the foreground."""
    from rltestbed.core.agents import default_params
    from rltestbed.core.envs import make_env
    from rltestbed.registry import create_run, update_run, keep_model, get_run
    from rltestbed.config import run_dir as run_dir_for

    params = dict(default_params(agent))
    for kv in param:
        if "=" not in kv:
            raise typer.BadParameter(f"--param expects key=value, got '{kv}'")
        k, v = kv.split("=", 1)
        if k not in params:
            raise typer.BadParameter(
                f"unknown hyperparameter '{k}' for agent '{agent}'"
            )
        default_typed = params[k]
        try:
            params[k] = (
                type(default_typed)(v)
                if not isinstance(default_typed, bool)
                else v.lower() in ("1", "true", "yes")
            )
        except ValueError:
            raise typer.BadParameter(f"cannot parse '{v}' for '{k}'")

    config = {
        "agent": agent,
        "env": env,
        "seed": seed,
        "episodes": episodes,
        "eval_freq": eval_freq,
        "max_steps": max_steps,
    }
    run_name = name or f"{agent}-{env}"
    run_id = create_run(run_name, agent, env, seed, episodes, params, config)

    typer.secho(
        f"[run {run_id}] {run_name}: {agent} on {env}, {episodes} episodes", fg="cyan"
    )

    update_run(run_id, status="running", started_at=time.time())
    from rltestbed.core.trainer import Trainer

    trainer = Trainer(
        run_id=run_id,
        run_dir=run_dir_for(run_id),
        env_name=env,
        agent_name=agent,
        params=params,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        eval_freq=eval_freq,
    )

    last_line = 0
    try:
        hist_path = Path(trainer.history_path)
        import threading

        def printer():
            pos = 0

            def drain():
                nonlocal pos
                if not hist_path.exists():
                    return
                with open(hist_path) as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
                for line in new.strip().splitlines():
                    rec = json.loads(line)
                    eval_txt = (
                        f" | eval {rec['eval']['episode_mean']:.1f}"
                        if "eval" in rec
                        else ""
                    )
                    metr = {
                        k[7:]: round(v, 4)
                        for k, v in rec.items()
                        if k.startswith("metric_")
                    }
                    typer.echo(
                        f"ep {rec['episode']:>5}/{episodes} | reward {rec['reward']:>8.2f} "
                        f"| mean100 {rec['mean_reward']:>8.2f}{eval_txt} {metr}"
                    )

            while True:
                alive = trainer_thread.is_alive()
                drain()
                if not alive:
                    time.sleep(0.5)
                    drain()
                    return
                time.sleep(0.4)

        trainer_thread = threading.Thread(target=lambda: trainer.run(), daemon=False)

        # signal handling stays on the main thread; forward Ctrl-C to trainer
        import signal as _signal

        def fwd(signum, frame):
            trainer.stop()

        _signal.signal(_signal.SIGINT, fwd)
        _signal.signal(_signal.SIGTERM, fwd)

        trainer_thread.start()
        printer()
        trainer_thread.join()
    finally:
        summary = trainer.summary()
        update_run(
            run_id,
            status=summary["status"] if summary["status"] != "stopping" else "stopped",
            finished_at=time.time(),
            episodes_done=summary["episodes_done"],
            final_mean_reward=summary["final_mean_reward"],
            best_eval_reward=summary["best_eval_reward"],
        )

    run = get_run(run_id)
    typer.secho(
        f"\n[{run['status']}] episodes={run['episodes_done']} "
        f"final_mean={run['final_mean_reward']} best_eval={run['best_eval_reward']}",
        fg="green",
    )
    if keep and run["best_eval_reward"] is not None:
        model = keep_model(run_id, Path(trainer.best_path))
        typer.secho(f"Model kept: id={model['id']} name={model['name']}", fg="green")
    typer.echo(f"Artifacts: {run_dir_for(run_id)}")


# ------------------------------------------------------------------ info --
@app.command(name="agents")
def agents_list():
    """List available agents with their hyperparameters."""
    from rltestbed.core.agents import list_agents

    for spec in list_agents():
        typer.secho(f"{spec['name']:<20} [{spec['type']}] {spec['label']}", bold=True)
        typer.echo(f"  {spec['description']}")
        ps = ", ".join(f"{p['name']}={p['default']}" for p in spec["params"])
        typer.echo(f"  params: {ps}\n")


@app.command(name="envs")
def envs_list():
    """List available environments."""
    from rltestbed.core.envs import list_envs

    rows = [
        (e["name"], e["category"], e["difficulty"], e["max_steps"], e["description"])
        for e in list_envs()
    ]
    width = max(len(r[0]) for r in rows) + 2
    for r in rows:
        typer.secho(f"{r[0]:<{width}}", nl=False, bold=True)
        typer.echo(f"{r[1]:<12} {r[2]:<14} steps~{r[3]:<5} {r[4]}")


@envs_app.command(name="detail")
def env_detail(name: str):
    """Show full metadata for one environment (probes spaces)."""
    from rltestbed.core.envs import get_env_spec

    spec = get_env_spec(name).to_dict(resolve=True)
    typer.echo(json.dumps(spec, indent=2))


# ------------------------------------------------------------------ runs --
@runs_app.command(name="list")
def runs_list(status: Optional[str] = None, limit: int = 50):
    from rltestbed.registry import list_runs

    for r in list_runs(limit=limit):
        if status and r["status"] != status:
            continue
        typer.echo(
            f"{r['id']}  {r['status']:<12} {r['agent']:<18} on {r['env']:<22} "
            f"eps={r['episodes_done']:>5}/{r['episodes']:<5} "
            f"best_eval={_num(r['best_eval_reward']):>8}  {_fmt_ts(r['created_at'])}  {r['name']}"
        )


def _num(x):
    return f"{x:.1f}" if isinstance(x, float) else str(x or "-")


@runs_app.command(name="show")
def runs_show(run_id: str):
    from rltestbed.registry import get_run

    r = get_run(run_id)
    if not r:
        raise typer.Exit(f"no such run: {run_id}")
    typer.echo(
        json.dumps(
            {**r, "config": {**r.get("config", {}), "params": r.get("params", {})}},
            indent=2,
            default=str,
        )
    )


@runs_app.command(name="delete")
def runs_delete(run_id: str, yes: bool = typer.Option(False, "--yes", "-y")):
    from rltestbed.registry import delete_run, get_run
    import shutil
    from rltestbed.config import run_dir as run_dir_for

    if not get_run(run_id):
        raise typer.Exit(f"no such run: {run_id}")
    if not yes:
        typer.confirm(f"Delete run {run_id} and its artifacts?", abort=True)
    shutil.rmtree(run_dir_for(run_id), ignore_errors=True)
    delete_run(run_id)
    typer.echo(f"deleted {run_id}")


# ---------------------------------------------------------------- models --
@models_app.command(name="list")
def models_list():
    from rltestbed.registry import list_models

    for m in list_models():
        typer.echo(
            f"{m['id']}  {m['name']:<40} {m['agent']:<16} on {m['env']:<18} "
            f"{m['size_bytes'] // 1024:>6} KB  best={_num(m['best_eval_reward']):>8}  "
            f"{_fmt_ts(m['created_at'])}"
        )


@models_app.command(name="delete")
def models_delete(model_id: str, yes: bool = typer.Option(False, "--yes", "-y")):
    from rltestbed.registry import delete_model, get_model

    if not get_model(model_id):
        raise typer.Exit(f"no such model: {model_id}")
    if not yes:
        typer.confirm(f"Delete model {model_id} from disk and registry?", abort=True)
    delete_model(model_id)
    typer.echo(f"deleted {model_id}")


@models_app.command(name="evaluate")
def models_evaluate(model_id: str, episodes: int = 20):
    """Evaluate a kept model greedily on its environment."""
    import numpy as np
    from rltestbed.core.agents import create_agent
    from rltestbed.core.envs import make_env
    from rltestbed.registry import get_model

    m = get_model(model_id)
    if not m:
        raise typer.Exit(f"no such model: {model_id}")
    env = make_env(m["env"])
    agent = create_agent(m["agent"], env.obs_dim, env.act_dim)
    agent.TRAIN = False
    agent.load_model(m["path"])
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        if not isinstance(obs, dict):
            obs = {
                "obs": list(np.atleast_1d(np.asarray(obs, dtype=float))),
                "action_mask": [1] * env.act_dim,
            }
        total, done, steps = 0.0, False, 0
        while not done and steps < 5000:
            a = agent.get_action(obs)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
            steps += 1
        rewards.append(total)
    env.close()
    typer.echo(
        json.dumps(
            {
                "model": m["name"],
                "env": m["env"],
                "episodes": episodes,
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
            },
            indent=2,
        )
    )


app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")
app.add_typer(envs_app, name="env-info")

@app.command(name="benchmarks")
def benchmarks_list():
    """List diagnostic benchmarks."""
    from rltestbed.benchmarks import benchmark_dict

    for b in benchmark_dict():
        typer.secho(f"{b['id']:<22} {b['title']:<35} probe={b['probe']}", bold=True)
        typer.echo(f"  {b['tests']}\n")

@app.command(name="compare")
def compare_runs(run_ids: str = typer.Argument(..., help="Comma-separated run ids")):
    """Compare multiple runs (prints summary table)."""
    from rltestbed.registry import get_run
    import json as _json
    from pathlib import Path as _P
    from rltestbed.config import run_dir as _rd

    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    for rid in ids:
        r = get_run(rid)
        if not r:
            typer.echo(f"{rid}: not found")
            continue
        hist_path = _rd(rid) / "history.jsonl"
        n = 0
        if hist_path.exists():
            n = sum(1 for _ in open(hist_path))
        typer.echo(f"{rid}  {r['agent']} on {r['env']}  status={r['status']}  best={r.get('best_eval_reward')}  history_lines={n}")




@app.command()
def play(
    env: str = typer.Argument(..., help="Environment to play (e.g. cartpole)"),
    model_id: str = typer.Option(None, "--model", help="Trained model ID (or omit for random)"),
    agent: str = typer.Option(None, help="Agent if no model (default random)"),
    episodes: int = typer.Option(1, help="Number of episodes to run"),
    render: str = typer.Option("human", help="Render mode: human (window) or rgb_array"),
    seed: int = typer.Option(42, help="Seed"),
    max_steps: int = typer.Option(500, help="Max steps per episode"),
):
    """Watch an agent play with true Gymnasium pygame rendering.

    Examples:
        rltestbed play cartpole                          # random on CartPole, window
        rltestbed play cartpole --model abcd1234         # your trained DQN, window
        rltestbed play gridworld --agent q_learning --model-id ...  # tabular
    """
    from rltestbed.core.envs import make_env, get_env_spec
    from rltestbed.core.agents import create_agent
    from rltestbed.registry import get_model
    import numpy as np, time

    spec = get_env_spec(env)
    # For gym with human, use gymnasium directly to get window
    is_gym = spec.category == 'gym'
    gym_env = None
    if is_gym and render == 'human':
        try:
            import gymnasium as gym
            gym_ids = {'cartpole': 'CartPole-v1', 'mountain_car': 'MountainCar-v0', 'lunar_lander': 'LunarLander-v2'}
            gid = gym_ids.get(env)
            if gid:
                gym_env = gym.make(gid, render_mode='human')
                gym_env.reset(seed=seed)
            else:
                gym_env = None
        except Exception as e:
            typer.echo(f"Could not create human window: {e}")
            gym_env = None

    # Fallback to adapter for obs handling
    wrap_env = make_env(env)
    # Resolve agent
    if model_id:
        m = get_model(model_id)
        if not m:
            raise typer.BadParameter(f"unknown model {model_id}")
        agent_name = m["agent"]
        # ensure env matches model's env if not explicitly overridden
        if m["env"] != env:
            typer.echo(f"Model was trained on {m['env']}, playing on {env} (obs dims may mismatch)")
        agent_obj = create_agent(agent_name, wrap_env.obs_dim, wrap_env.act_dim)
        agent_obj.load_model(m["path"])
        agent_obj.TRAIN = False
        typer.echo(f"Loaded model {model_id} ({agent_name})")
    else:
        agent_name = agent or "random"
        if agent_name == "random":
            from rltestbed.core.agents.random_agent import RandomAgent
            agent_obj = RandomAgent(wrap_env.obs_dim, wrap_env.act_dim, {})
        else:
            agent_obj = create_agent(agent_name, wrap_env.obs_dim, wrap_env.act_dim)
            agent_obj.TRAIN = False

    for ep in range(episodes):
        obs = wrap_env.reset() if not gym_env else gym_env.reset(seed=seed+ep)[0] if isinstance(gym_env.reset(seed=seed+ep), tuple) else gym_env.reset(seed=seed+ep)
        # Normalize obs for adapter vs gym
        if is_gym and gym_env:
            # convert like GymAdapter
            import gymnasium.spaces as spaces
            # use adapter's convert logic
            if isinstance(obs, (int, np.integer)):
                vec = np.zeros(wrap_env.obs_dim, dtype=np.float32); vec[int(obs)]=1.0; obs = vec.tolist()
            else:
                obs = np.asarray(obs, dtype=np.float32).flatten().tolist()
            if spec.action_masked:
                obs = {"obs": obs, "action_mask": [1]*wrap_env.act_dim}
        # Gym human needs initial render
        if gym_env and render=="human":
            try: gym_env.render()
            except: pass
        total=0
        done=False
        steps=0
        while not done and steps < max_steps:
            # get action
            if agent_name=="random":
                _, mask = agent_obj._unpack(obs) if hasattr(agent_obj,'_unpack') else (None,None)
                if mask:
                    valid=[i for i,v in enumerate(mask) if v]
                    action=int(np.random.choice(valid)) if valid else 0
                else:
                    action=int(np.random.randint(0, wrap_env.act_dim))
            else:
                action=int(agent_obj.get_action(obs))
            # step
            if gym_env and render=="human":
                try:
                    obs_raw, reward, term, trunc, info = gym_env.step(int(action))
                    # convert again
                    if isinstance(obs_raw, (int, np.integer)):
                        vec=np.zeros(wrap_env.obs_dim, dtype=np.float32); vec[int(obs_raw)]=1.0; obs=vec.tolist()
                    else:
                        obs=np.asarray(obs_raw, dtype=np.float32).flatten().tolist()
                    if spec.action_masked:
                        obs={"obs":obs,"action_mask":[1]*wrap_env.act_dim}
                    term, trunc = bool(term), bool(trunc)
                    reward=float(reward)
                    try: gym_env.render()
                    except: pass
                    time.sleep(0.02)
                except Exception as e:
                    typer.echo(f"step failed: {e}"); break
            else:
                try:
                    obs, reward, term, trunc, info = wrap_env.step(int(action))
                except:
                    nxt=wrap_env.step(int(action))
                    if len(nxt)==5:
                        obs,reward,term,trunc,info=nxt
                    else:
                        obs,reward,term,trunc,info=nxt[0],0,True,False,{}
                reward=float(reward); term=bool(term); trunc=bool(trunc)
                if render=="human" and hasattr(wrap_env,'_env') and hasattr(wrap_env._env,'render'):
                    try: wrap_env._env.render()
                    except: pass
            total+=reward
            done=term or trunc
            steps+=1
            typer.echo(f"ep {ep} step {steps} act {action} r {reward:.2f} total {total:.1f} {'done' if done else ''}")
            if steps>=max_steps:
                break
        typer.secho(f"Episode {ep} finished: {steps} steps, total {total:.1f}", fg="green")
        if gym_env and render=="human":
            time.sleep(0.5)
    try:
        if gym_env: gym_env.close()
        else: wrap_env.close()
    except: pass


# ------------------------------------------------------- worker (internal) --
@app.command(name="train-worker", hidden=True)
def train_worker(
    run_id: str = typer.Option(...),
):
    """Internal: executed by the job manager as a separate process."""
    from rltestbed.registry import get_run, update_run
    from rltestbed.config import run_dir as run_dir_for
    from rltestbed.core.trainer import Trainer

    run = get_run(run_id)
    if not run:
        raise typer.BadParameter(f"unknown run '{run_id}'")
    config = run.get("config") or {}
    update_run(
        run_id, status="running", pid=__import__("os").getpid(), started_at=time.time()
    )
    import traceback

    try:
        trainer = _build_trainer(Trainer, run_id, run, config, run_dir_for)
    except Exception as exc:
        # Construction failures (missing dependency, bad params) must still leave
        # a terminal status and a readable reason behind.
        log = run_dir_for(run_id)
        log.mkdir(parents=True, exist_ok=True)
        with open(log / "worker.log", "a") as fh:
            fh.write(f"Run could not start: {type(exc).__name__}: {exc}\n")
            fh.write(traceback.format_exc())
        (log / "progress.json").write_text(json.dumps(
            {"status": "failed", "episode": 0, "error": f"{type(exc).__name__}: {exc}"}))
        update_run(run_id, status="failed", finished_at=time.time(), pid=None)
        raise typer.Exit(1)

    try:
        summary = trainer.run()
    except Exception:
        traceback.print_exc()
        update_run(run_id, status="failed", finished_at=time.time(), pid=None)
        raise
    update_run(
        run_id,
        status=summary["status"],
        finished_at=time.time(),
        episodes_done=summary["episodes_done"],
        final_mean_reward=summary["final_mean_reward"],
        best_eval_reward=summary["best_eval_reward"],
        pid=None,
    )


def _build_trainer(Trainer, run_id, run, config, run_dir_for):
    return Trainer(
        run_id=run_id,
        run_dir=run_dir_for(run_id),
        env_name=run["env"],
        agent_name=run["agent"],
        params=run.get("params") or {},
        seed=run["seed"],
        episodes=run["episodes"],
        max_steps=config.get("max_steps"),
        eval_freq=config.get("eval_freq", 25),
        eval_episodes=config.get("eval_episodes", 10),
        checkpoint_freq=config.get("checkpoint_freq", 100),
        device=run.get("device", "cpu"),
        save_best=config.get("save_best", True),
        save_final=config.get("save_final", True),
        auto_eval=config.get("auto_eval", True),
        early_stopping_patience=config.get("early_stopping_patience"),
        log_wandb=config.get("log_wandb", False),
    )


if __name__ == "__main__":
    app()
