"""What the numbers in an observation mean, for the Visualize page.

Labels come from Gymnasium's documentation for the classic tasks and are
derived from the physics model's own joint names for MuJoCo, so they stay
correct if a model changes. Where no per-number label exists, ``describe``
still explains what the vector contains.
"""

from typing import Any, Dict, List, Optional

CLASSIC: Dict[str, List[str]] = {
    "cartpole": ["cart position", "cart velocity", "pole angle (rad)", "pole angular velocity"],
    "mountain_car": ["car position", "car velocity"],
    "mountain_car_continuous": ["car position", "car velocity"],
    "acrobot": ["cos θ1", "sin θ1", "cos θ2", "sin θ2", "θ1 angular velocity",
                "θ2 angular velocity"],
    "pendulum": ["cos θ", "sin θ", "angular velocity"],
    "lunar_lander": ["x", "y", "x velocity", "y velocity", "angle", "angular velocity",
                     "left leg contact", "right leg contact"],
    "catch": ["paddle x (0–1)", "ball x (0–1)", "ball y (0–1)"],
}

DESCRIPTIONS: Dict[str, str] = {
    "bipedal_walker": "24 numbers: hull angle and velocities, 4 joint angles and speeds, "
    "2 leg-contact flags, and 10 lidar distances ahead of the walker.",
    "inverted_double_pendulum": "11 numbers: cart position, sines and cosines of both hinge "
    "angles, velocities, and constraint forces.",
    "reacher": "11 numbers: cosines and sines of both arm joints, the target position, "
    "joint velocities, and the fingertip-to-target vector (x, y, z).",
    "pusher": "23 numbers: 7 arm joint angles and velocities, and the positions of the "
    "fingertip, the object and the goal.",
    "humanoid": "376 numbers: joint positions and velocities plus body inertias, velocities "
    "and external contact forces for each body part.",
    "humanoid_standup": "376 numbers, laid out exactly as Humanoid's.",
    "blackjack": "A one-hot encoding of (your hand total, the dealer's showing card, whether "
    "you hold a usable ace).",
    "texas_holdem": "72 numbers: one-hot encodings of the cards you can see and the chips "
    "raised in each betting round.",
}


def mujoco_labels(env: Any, obs_dim: int) -> Optional[List[str]]:
    """Labels for MuJoCo observations of the form qpos[k:] + qvel.

    Built from the model's joint names and types. Returns None when the
    observation has a different layout (it then gets a text description).
    """
    try:
        import mujoco

        model = env.unwrapped.model
    except Exception:
        return None
    qpos: List[str] = []
    qvel: List[str] = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        jtype = int(model.jnt_type[j])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos += [f"{name} {a}" for a in ("x", "y", "z", "qw", "qx", "qy", "qz")]
            qvel += [f"{name} {a} velocity" for a in ("x", "y", "z", "roll", "pitch", "yaw")]
        elif jtype == int(mujoco.mjtJoint.mjJNT_BALL):
            qpos += [f"{name} {a}" for a in ("qw", "qx", "qy", "qz")]
            qvel += [f"{name} {a} velocity" for a in ("x", "y", "z")]
        else:  # hinge or slide: one coordinate each
            qpos.append(f"{name} {'angle' if jtype == int(mujoco.mjtJoint.mjJNT_HINGE) else 'position'}")
            qvel.append(f"{name} velocity")
    skip = len(qpos) + len(qvel) - obs_dim
    # Locomotion tasks hide the absolute x (and y) position from the agent.
    if 0 <= skip <= 2:
        return qpos[skip:] + qvel
    return None


def labels_for(name: str, env: Any, obs_dim: int) -> Optional[List[str]]:
    if name in CLASSIC and len(CLASSIC[name]) == obs_dim:
        return CLASSIC[name]
    inner = getattr(env, "_env", None)
    if inner is not None and hasattr(getattr(inner, "unwrapped", None), "model"):
        return mujoco_labels(inner, obs_dim)
    return None


def explain(source: str, spec: Any, labels: Optional[List[str]], render_error: str = "") -> str:
    """One paragraph: what the picture is, where it comes from, and why."""
    obs_text = (
        f"The panel lists all {spec.obs_dim} numbers the agent receives, labelled."
        if labels else
        DESCRIPTIONS.get(spec.name)
        or (f"The agent receives a one-hot vector: a single 1 marking which of {spec.obs_dim} "
            "states it is in." if spec.obs_type == "onehot" else
            f"The panel lists the {spec.obs_dim} numbers the agent receives.")
    )
    if source == "pixels":
        return (
            "Real frames from the environment's own renderer, taken from the same "
            "environment instance the agent is acting in, so the picture and the numbers "
            "always describe the same moment. The agent never sees these pixels: "
            + obs_text[0].lower() + obs_text[1:]
        )
    if source == "frames":
        return (
            "The agent's actual input, drawn as an image: each colour is one channel of "
            "the 10×10 feature grid (player, enemies, trails and so on). Unlike most "
            "environments here, what you see is exactly what the agent sees."
        )
    if source == "schematic":
        return (
            f"{spec.label} has no picture of its own, so this is a drawing of its internal "
            "state: where the agent is and what surrounds it. Anything shown that the agent "
            "cannot observe (such as true reward values) is labelled as hidden. " + obs_text
        )
    reason = f" Rendering was unavailable here: {render_error}" if render_error else ""
    return (
        f"{spec.label} has nothing meaningful to draw: its whole state is the handful of "
        "numbers below, which is exactly what the agent sees. " + obs_text + reason
    )
