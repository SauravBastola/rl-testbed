"""Visualize rollouts: picture sources, labels, final states, GL backend choice,
the isolated worker, and upstream renderer workarounds."""

import importlib.util
import sys

import pytest


def _run(env, steps=6, **kw):
    from rltestbed import rollout

    return rollout.run({"env": env, "agent": "random", "seed": 0, "max_steps": steps, **kw})


def test_builtin_worlds_are_drawn_from_their_state():
    r = _run("dyna_maze")
    assert r["render"]["source"] == "schematic"
    steps = r["trajectories"][0]["steps"]
    assert all(s["view"] and s["view"]["kind"] == "grid" for s in steps)
    assert steps[0]["view"]["agent"] == steps[0]["view"]["start"]  # pictured before acting


def test_every_episode_ends_with_its_final_state():
    traj = _run("gridworld", steps=5)["trajectories"][0]
    *acted, last = traj["steps"]
    assert len(acted) == traj["length"] and last["final"] is True and last["action"] is None
    assert all(s["action"] is not None for s in acted)


def test_bandit_view_shows_hidden_true_values():
    view = _run("multi_armed_bandit")["trajectories"][0]["steps"][-1]["view"]
    assert view["kind"] == "bandit" and len(view["values"]) == 10 and view["last"] is not None


def test_probes_have_nothing_to_draw_and_say_so():
    r = _run("probe2")
    assert r["render"]["source"] == "observation"
    assert "nothing meaningful to draw" in r["render"]["explanation"]


def test_classic_control_renders_pixels_with_labels():
    pytest.importorskip("pygame")
    r = _run("cartpole")
    assert r["render"]["source"] == "pixels" and r["render"]["mime"] == "image/png"
    assert all(s["frame"] for s in r["trajectories"][0]["steps"])
    assert r["obs_labels"][2] == "pole angle (rad)"


def test_pendulum_renders_every_frame():
    """Gymnasium 0.29 + NumPy 2: Pendulum's arrow size was a numpy scalar pygame rejected."""
    pytest.importorskip("pygame")
    r = _run("pendulum", steps=15)
    assert r["render"]["error"] is None
    assert all(s["frame"] for s in r["trajectories"][0]["steps"])


def test_unknown_environment_is_a_404():
    from rltestbed import rollout

    with pytest.raises(rollout.RolloutError) as err:
        rollout.run({"env": "nope"})
    assert err.value.status == 404


def test_only_mujoco_needs_a_subprocess():
    from rltestbed import rollout

    assert rollout.needs_isolation("half_cheetah")
    assert not rollout.needs_isolation("cartpole") and not rollout.needs_isolation("gridworld")


@pytest.mark.parametrize("platform,display,explicit,expected", [
    ("darwin", None, None, None),     # macOS: GLFW on the subprocess main thread
    ("win32", None, None, None),
    ("linux", ":0", None, None),      # a desktop session: GLFW
    ("linux", None, None, "egl"),     # headless Linux: EGL
    ("darwin", None, "osmesa", "osmesa"),  # an explicit choice always wins
])
def test_gl_backend_choice(monkeypatch, platform, display, explicit, expected):
    from rltestbed import rollout

    monkeypatch.setattr(sys, "platform", platform)
    for var, val in (("DISPLAY", display), ("MUJOCO_GL", explicit)):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)
    assert rollout.mujoco_gl_backend() == expected


def test_isolated_worker_round_trip():
    """The subprocess path end to end, on a cheap environment."""
    from rltestbed import rollout

    r = rollout.run_isolated({"env": "gridworld", "agent": "random", "seed": 1, "max_steps": 4})
    assert r["render"]["source"] == "schematic" and len(r["trajectories"][0]["steps"]) >= 2
    with pytest.raises(rollout.RolloutError) as err:
        rollout.run_isolated({"env": "nope"})
    assert err.value.status == 404


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="MuJoCo not installed")
def test_mujoco_labels_come_from_the_model():
    from rltestbed.core.envs.adapters import GymAdapter
    from rltestbed.core.envs.observations import labels_for

    env = GymAdapter("HalfCheetah-v4")
    labels = labels_for("half_cheetah", env, env.obs_dim)
    assert len(labels) == 17 and labels[0] == "rootz position" and labels[2] == "bthigh angle"


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="MuJoCo not installed")
def test_failed_viewer_cleanup_is_silent():
    """A viewer whose OpenGL context never existed must not raise on cleanup."""
    import rltestbed  # noqa: F401  applies the compat shims
    from gymnasium.envs.mujoco import mujoco_rendering as mr

    viewer = mr.OffScreenViewer.__new__(mr.OffScreenViewer)
    viewer.free()  # used to raise AttributeError: no attribute 'opengl_context'


def test_minatar_frames_are_scaled_and_colour_coded():
    pytest.importorskip("minatar")
    import base64
    import io

    from PIL import Image

    r = _run("minatar_breakout", steps=3, episodes=3)
    assert len(r["trajectories"]) == 3  # several episodes in one request
    names = [x["name"] for x in r["render"]["legend"]]
    assert names == ["paddle", "ball", "trail", "brick"]
    frame = r["trajectories"][0]["steps"][0]["frame"]
    assert Image.open(io.BytesIO(base64.b64decode(frame))).size == (320, 320)  # was 10x10
