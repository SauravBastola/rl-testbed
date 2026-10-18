"""Play episodes for the Visualize page.

Each step records the state the agent acted in (picture or schematic, plus the
observation it received), the action it chose and the reward it got, then a
final record for the end state. Pictures come from the *same* environment
instance the agent acts in, so frames and numbers always agree.

What a step's picture is depends on the environment (``render.source``):

- ``pixels``     the environment's own renderer (Gymnasium, MuJoCo, PettingZoo)
- ``frames``     MinAtar's feature grid, which is also exactly what the agent sees
- ``schematic``  builtin worlds with no renderer describe their state as data
                 (``view()``), drawn by the web app in the current theme
- ``observation`` nothing to draw: the observation vector is the whole state

MuJoCo needs an OpenGL context. On macOS the default backend (GLFW) only works
on a process's main thread, and web requests run on worker threads, so MuJoCo
rollouts run in a short-lived subprocess (:func:`run_isolated`).
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import numpy as np

MAX_EPISODES = 20
MAX_STEPS = 2000
MAX_SIDE = 480  # px; keeps a 500-step rollout to a reasonable payload


class RolloutError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def needs_isolation(env_name: str) -> bool:
    from rltestbed.core.envs import REGISTRY

    entry = REGISTRY.get(env_name)
    return bool(entry) and entry["spec"].category == "mujoco"


def mujoco_gl_backend() -> Optional[str]:
    """The OpenGL backend for a rendering subprocess, or None for MuJoCo's default.

    macOS and Windows: the default (GLFW/WGL) works on the subprocess's main
    thread. Linux without a display: EGL (GPU or Mesa), the usual headless choice.
    An explicit MUJOCO_GL in the environment always wins.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return "egl"
    return None


def _make(env_name: str, spec: Any, seed: int):
    """Build the environment, preferring a renderable instance. Returns (env, source, error)."""
    from rltestbed.core.envs import make_env
    from rltestbed.core.envs.adapters import GymAdapter, PettingZooSelfPlayAdapter
    from rltestbed.core.envs.renderers import ENV_IDS

    if env_name in ENV_IDS:
        try:
            return GymAdapter(ENV_IDS[env_name], render_mode="rgb_array"), "pixels", ""
        except Exception as exc:  # renderer unavailable: still play the episode
            return make_env(env_name), "observation", f"{type(exc).__name__}: {exc}"
    if spec.category == "pettingzoo":
        try:
            return PettingZooSelfPlayAdapter(env_name, seed=seed, render_mode="rgb_array"), "pixels", ""
        except Exception as exc:
            return make_env(env_name), "observation", f"{type(exc).__name__}: {exc}"
    env = make_env(env_name)
    if spec.category == "minatar":
        return env, "frames", ""
    if hasattr(getattr(env, "_env", env), "view"):
        return env, "schematic", ""
    return env, "observation", ""


def _encode(arr, photographic: bool) -> Optional[str]:
    """Encode a frame quickly.

    3-D scenes (MuJoCo) are smooth gradients: JPEG is ~40x faster to encode than
    an optimised PNG and visually identical at this size. Pixel art (MinAtar,
    toy text, board games) stays PNG, with fast compression, so edges stay crisp.
    """
    import base64
    import io

    from PIL import Image

    from rltestbed.core.envs.renderers import encode_array

    if not photographic:
        # encode_array handles layouts/dtypes; re-encode its result without optimisation
        png = encode_array(arr, max_size=(MAX_SIDE, MAX_SIDE))
        return png
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    image = Image.fromarray(a[..., :3])
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _before_render(target) -> None:
    """Work around upstream renderer quirks before calling ``render()``.

    Gymnasium 0.29's Pendulum sizes its torque arrow from ``last_u``; under
    NumPy 2 that is a numpy scalar, which pygame rejects ("size must be two
    numbers"), so every frame after the first failed. Coerce it to a float.
    """
    inner = getattr(target, "unwrapped", None)
    u = getattr(inner, "last_u", None)
    if isinstance(u, (np.generic, np.ndarray)):
        inner.last_u = float(np.asarray(u).reshape(-1)[0])


def _flat(obs) -> list:
    raw = obs.get("obs", []) if isinstance(obs, dict) else obs
    return np.asarray(raw, dtype=float).flatten().tolist()


def run(body: Dict[str, Any]) -> Dict[str, Any]:
    """Play episodes and return trajectories. Raises RolloutError."""
    from rltestbed.core.agents import create_agent, list_agents
    from rltestbed.core.envs import REGISTRY
    from rltestbed.core.envs.observations import explain, labels_for
    from rltestbed.core.envs.renderers import encode_array
    from rltestbed.registry import get_model

    env_name = body.get("env")
    agent_name = body.get("agent")
    model_id = body.get("model_id")
    seed = int(body.get("seed", 42))
    max_steps = max(1, min(int(body.get("max_steps", 300)), MAX_STEPS))
    episodes = max(1, min(int(body.get("episodes", 1)), MAX_EPISODES))

    model = None
    if model_id:
        model = get_model(model_id)
        if not model:
            raise RolloutError(404, f"unknown model '{model_id}'")
        agent_name, env_name = model["agent"], model["env"]
    agent_name = agent_name or "random"
    if not env_name or env_name not in REGISTRY:
        raise RolloutError(404, f"unknown env '{env_name}'")
    spec = REGISTRY[env_name]["spec"]
    if not any(a["name"] == agent_name for a in list_agents()):
        raise RolloutError(422, f"unknown agent '{agent_name}'")

    np.random.seed(seed)
    try:
        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
    except Exception:
        pass

    env, source, render_error = _make(env_name, spec, seed)
    photographic = spec.category == "mujoco"
    try:
        agent = create_agent(agent_name, env.obs_dim, env.act_dim, device="cpu")
        if spec.action_type == "continuous" and getattr(env, "act_low", None) is not None:
            agent.set_action_bounds(env.act_low, env.act_high)
        if model:
            agent.load_model(model["path"])
        agent.TRAIN = False
    except Exception as exc:
        env.close()
        raise RolloutError(422, f"cannot prepare the agent: {exc}") from exc

    def picture() -> Dict[str, Any]:
        """The current state as a frame (base64 PNG) and/or a schematic view."""
        nonlocal source, render_error
        out: Dict[str, Any] = {"frame": None, "view": None}
        try:
            if source == "pixels":
                target = env._env if hasattr(env, "_env") and hasattr(env._env, "render") else env
                _before_render(target)
                out["frame"] = _encode(target.render(), photographic)
            elif source == "frames":
                out["frame"] = encode_array(env._env.state(), max_size=(MAX_SIDE, MAX_SIDE))
            elif source == "schematic":
                out["view"] = getattr(env, "_env", env).view()
        except Exception as exc:
            # A renderer failing mid-episode must not lose the episode.
            render_error = render_error or f"{type(exc).__name__}: {exc}"
            source = "observation"
        return out

    trajectories = []
    for ep in range(episodes):
        try:
            obs = env.reset(seed=seed + ep)
        except TypeError:  # adapters without a seed argument
            obs = env.reset()
        steps, total, done, s = [], 0.0, False, 0
        while not done and s < max_steps:
            pic = picture()
            action = agent.get_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            steps.append({
                "step": s, "obs": _flat(obs),
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
                "reward": float(reward), "total_reward": total,
                "terminated": bool(term), "truncated": bool(trunc), **pic,
            })
            obs, done, s = next_obs, bool(term or trunc), s + 1
        # the state the episode ended in, so the last picture is not missing
        steps.append({"step": s, "obs": _flat(obs), "action": None, "reward": 0.0,
                      "total_reward": total, "terminated": done, "truncated": not done,
                      "final": True, **picture()})
        trajectories.append({"episode": ep, "steps": steps, "total_reward": total,
                             "length": s})
    labels = labels_for(env_name, env, env.obs_dim)
    try:
        env.close()
    except Exception:
        pass
    return {
        "env": env_name, "agent": agent_name, "model_id": model_id, "seed": seed,
        "render": {"source": source, "error": render_error or None,
                   "mime": "image/jpeg" if photographic and source == "pixels" else "image/png",
                   "explanation": explain(source, spec, labels, render_error)},
        "obs_labels": labels, "obs_type": spec.obs_type,
        "trajectories": trajectories,
    }


def run_isolated(body: Dict[str, Any], timeout: float = 240.0) -> Dict[str, Any]:
    """Run :func:`run` in a subprocess (its main thread owns the GL context)."""
    with tempfile.TemporaryDirectory(prefix="rltestbed-rollout-") as tmp:
        req, out = os.path.join(tmp, "request.json"), os.path.join(tmp, "result.json")
        with open(req, "w") as fh:
            json.dump(body, fh)
        env = dict(os.environ)
        backend = mujoco_gl_backend()
        if backend:
            env["MUJOCO_GL"] = backend
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "rltestbed.cli", "rollout-worker", req, out],
                env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RolloutError(504, f"rendering took longer than {int(timeout)}s") from exc
        if os.path.exists(out):
            with open(out) as fh:
                result = json.load(fh)
            if "error" in result:
                raise RolloutError(result.get("status", 500), result["error"])
            return result
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise RolloutError(500, "the rendering process exited without a result: " + " | ".join(tail))
