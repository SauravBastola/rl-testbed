"""Environment rendering registry used by the Visualize page.

Rendering is a presentation concern, not a training concern. Every registered
environment therefore gets a frame even when its upstream package does not
ship a renderer or a headless OpenGL backend is unavailable:

1. Prefer a real Gymnasium ``rgb_array`` frame when one can be created.
2. Fall back to a native adapter frame (MinAtar state / builtin ASCII).
3. Fall back to a consistent observation dashboard: vector bars, action,
   reward and step context. This is intentionally useful rather than a blank
   placeholder — it makes *every* environment inspectable.

All functions return base64-encoded PNG strings so the existing API contract
(``step.frame``) stays unchanged.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Gymnasium IDs for every registered Gym-family environment. Keep this mapping
# here instead of coupling the API to the large registry module.
ENV_IDS: Dict[str, str] = {
    # classic / Box2D
    "cartpole": "CartPole-v1",
    "mountain_car": "MountainCar-v0",
    "lunar_lander": "LunarLander-v2",
    "acrobot": "Acrobot-v1",
    "pendulum": "Pendulum-v1",
    "bipedal_walker": "BipedalWalker-v3",
    # continuous classic
    "mountain_car_continuous": "MountainCarContinuous-v0",
    # toy text
    "frozen_lake": "FrozenLake-v1",
    "cliff_walking": "CliffWalking-v0",
    "taxi": "Taxi-v3",
    "blackjack": "Blackjack-v1",
    # MuJoCo
    "inverted_pendulum": "InvertedPendulum-v4",
    "inverted_double_pendulum": "InvertedDoublePendulum-v4",
    "reacher": "Reacher-v4",
    "pusher": "Pusher-v4",
    "half_cheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
    "swimmer": "Swimmer-v4",
    "ant": "Ant-v4",
    "humanoid": "Humanoid-v4",
    "humanoid_standup": "HumanoidStandup-v4",
}


def make_render_env(name: str):
    """Create a fresh rgb-array environment for a Gym-family name.

    Return ``None`` when the upstream renderer cannot initialise. The caller
    should still use :func:`render_frame`, which always creates a generic
    observation frame.
    """
    env_id = ENV_IDS.get(name)
    if not env_id:
        return None
    try:
        # Never set MUJOCO_GL here: it is process-wide and EGL does not exist on
        # macOS. rltestbed/rollout.py chooses a backend per rendering subprocess.
        import gymnasium as gym

        return gym.make(env_id, render_mode="rgb_array")
    except Exception:
        try:
            # A second attempt allows Gymnasium to select its platform default.
            import gymnasium as gym

            return gym.make(env_id, render_mode="rgb_array")
        except Exception:
            return None


def _png_base64(image) -> Optional[str]:
    try:
        buf = io.BytesIO()
        # optimize=True costs ~45 ms per frame for little size gain
        image.save(buf, format="PNG", compress_level=3)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def encode_array(arr: Any, max_size=(640, 420)) -> Optional[str]:
    """Encode HWC/CHW grayscale/RGB/RGBA arrays as a PNG."""
    try:
        from PIL import Image

        if arr is None:
            return None
        try:
            import torch

            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return None
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr)
            if arr.max(initial=0) <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            image = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                image = Image.fromarray(arr[..., 0], mode="L").convert("RGB")
            else:
                image = Image.fromarray(arr[..., :3], mode="RGB")
        else:
            return None
        image.thumbnail(max_size, Image.Resampling.NEAREST)
        return _png_base64(image)
    except Exception:
        return None


def _unwrap(obs: Any):
    if isinstance(obs, dict):
        return obs.get("obs", obs.get("observation", []))
    return obs


def _generic_frame(name: str, obs: Any, action: Any, reward: float, step: int,
                   label: str = "Observation view") -> Optional[str]:
    """Draw a useful, theme-neutral observation dashboard for any env."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        W, H = 640, 360
        bg = (10, 11, 15)
        panel = (20, 22, 29)
        border = (53, 59, 70)
        text = (236, 238, 242)
        dim = (156, 163, 175)
        brand = (168, 85, 247)
        cyan = (34, 211, 238)
        green = (163, 230, 53)

        image = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(image)
        try:
            mono = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 14)
            small = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 11)
            title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        except Exception:
            mono = small = title = ImageFont.load_default()

        d.rounded_rectangle([14, 14, W - 14, H - 14], radius=16, fill=panel, outline=border, width=1)
        d.text((32, 28), str(label), fill=dim, font=small)
        d.text((32, 48), str(name).replace("_", " ").title(), fill=text, font=title)
        d.text((W - 32, 32), f"STEP {step:04d}", fill=dim, font=small, anchor="ra")

        values = np.asarray(_unwrap(obs), dtype=np.float32).reshape(-1)
        values = np.nan_to_num(values)
        if values.size == 0:
            values = np.zeros(1, dtype=np.float32)
        # Show first 32 features, normalised only for display.
        shown = values[:32]
        max_abs = float(np.max(np.abs(shown))) or 1.0
        d.text((32, 92), f"observation  dim={values.size}", fill=dim, font=small)
        left, top = 32, 116
        bar_w, gap, row_h = 17, 3, 42
        for i, value in enumerate(shown):
            x = left + i * (bar_w + gap)
            if x + bar_w > W - 32:
                break
            norm = float(value / max_abs)
            y0 = top + row_h
            y1 = y0 - int(norm * 28)
            d.line([x + bar_w / 2, top, x + bar_w / 2, y0], fill=(53, 59, 70), width=1)
            color = cyan if norm >= 0 else brand
            d.rounded_rectangle([x, min(y1, y0), x + bar_w, max(y1, y0)], radius=3, fill=color)
            if i < 12:
                d.text((x + bar_w / 2, y0 + 5), str(i), fill=dim, font=small, anchor="ma")
        if values.size > 32:
            d.text((W - 32, top + 35), f"+{values.size - 32} more", fill=dim, font=small, anchor="ra")

        d.line([32, 198, W - 32, 198], fill=border, width=1)
        d.text((32, 218), "ACTION", fill=dim, font=small)
        d.text((32, 239), str(np.asarray(action).reshape(-1).tolist() if isinstance(action, (list, tuple, np.ndarray)) else action), fill=text, font=mono)
        d.text((W - 32, 218), "REWARD", fill=dim, font=small, anchor="ra")
        d.text((W - 32, 239), f"{float(reward):+.3f}", fill=green if float(reward) >= 0 else (251, 113, 133), font=mono, anchor="ra")
        d.text((32, 310), "Generic renderer · full observation inspectable", fill=dim, font=small)
        return _png_base64(image)
    except Exception:
        return None


def render_frame(name: str, adapter: Any, render_env: Any, obs: Any,
                 action: Any, reward: float, step: int = 0) -> Optional[str]:
    """Return the best available frame for any registered environment."""
    # 1. Real upstream rgb_array — MuJoCo, Gym classic, toy-text where supported.
    if render_env is not None:
        try:
            frame = encode_array(render_env.render())
            if frame:
                return frame
        except Exception:
            pass

    # 2. MinAtar's compact state array.
    if name.startswith("minatar"):
        try:
            state = adapter._env.state()
            frame = encode_array(state)
            if frame:
                return frame
        except Exception:
            pass

    # 3. Native ASCII builtin render as a readable image.
    try:
        raw = getattr(adapter, "_env", adapter)
        native = raw.render() if hasattr(raw, "render") else None
        if isinstance(native, str):
            from PIL import Image, ImageDraw, ImageFont

            image = Image.new("RGB", (640, 360), (10, 11, 15))
            d = ImageDraw.Draw(image)
            try:
                f = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 14)
            except Exception:
                f = ImageFont.load_default()
            d.rounded_rectangle([14, 14, 626, 346], radius=16, fill=(20, 22, 29), outline=(53, 59, 70))
            for i, line in enumerate(native.splitlines()[:18]):
                d.text((32, 30 + i * 17), line[:78], fill=(236, 238, 242) if i == 0 else (156, 163, 175), font=f)
            return _png_base64(image)
    except Exception:
        pass

    # 4. Universal final fallback. Never return a blank frame.
    return _generic_frame(name, obs, action, reward, step)


__all__ = ["ENV_IDS", "make_render_env", "render_frame", "encode_array"]
