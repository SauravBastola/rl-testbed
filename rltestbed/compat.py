"""Runtime compatibility shims for pinned third-party versions.

The testbed pins ``gymnasium==0.29.1`` because MinAtar, PettingZoo and the
existing custom adapters are verified against it. Newer releases of its
*dependencies* have since moved on, which leaves two real breakages that would
otherwise make whole environment families unusable:

1. ``numpy>=2.0`` removed ``np.float_``.  ``gymnasium/envs/classic_control/
   acrobot.py`` still uses it inside its RK4 integrator, so every ``step()``
   raises ``AttributeError``.

2. ``mujoco>=3.0`` rejects the legacy ``collision="predefined"`` attribute that
   gymnasium 0.29.1 still ships in some bundled XML assets (``swimmer.xml``).
   MuJoCo refuses to compile the model at all.

Both are fixed here at runtime instead of by editing ``site-packages``, so the
patches survive reinstalls, are reversible, and are clearly auditable.  Every
helper is idempotent and safe to call repeatedly.

Call :func:`apply_compat_shims` once at process start (the CLI, the server and
the trainer all do this).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

_APPLIED = False

# Attributes MuJoCo 3.x dropped from its MJCF schema but that still appear in
# the XML assets bundled with gymnasium 0.29.1.
_LEGACY_MJCF_ATTRS = ("collision",)

_XML_CACHE_DIR = Path(tempfile.gettempdir()) / "rltestbed-mujoco-assets"


def patch_numpy_aliases() -> bool:
    """Restore ``np.float_`` (and friends) removed in NumPy 2.0.

    Several pinned libraries still reference these deprecated aliases.  Adding
    them back is harmless for NumPy 1.x where they already exist.
    """
    import numpy as np

    changed = False
    for alias, target in (
        ("float_", np.float64),
        ("complex_", np.complex128),
        ("unicode_", np.str_),
        ("string_", np.bytes_),
    ):
        if not hasattr(np, alias):
            setattr(np, alias, target)
            changed = True
    return changed


def _sanitize_mjcf(xml_text: str) -> str:
    """Strip MuJoCo-3-incompatible attributes from an MJCF document."""
    for attr in _LEGACY_MJCF_ATTRS:
        xml_text = re.sub(rf'\s+{attr}="[^"]*"', "", xml_text)
    return xml_text


def _sanitized_asset(asset_path: Path) -> Path:
    """Return a path to a sanitized copy of ``asset_path`` (cached on disk)."""
    _XML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _XML_CACHE_DIR / asset_path.name
    if not out.exists() or out.stat().st_mtime < asset_path.stat().st_mtime:
        text = asset_path.read_text(encoding="utf-8", errors="replace")
        sanitized = _sanitize_mjcf(text)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(sanitized, encoding="utf-8")
        tmp.replace(out)
    return out


def patch_mujoco_assets() -> int:
    """Rewrite legacy MJCF assets and redirect ``MujocoEnv`` to the copies.

    Returns the number of asset files that needed patching.
    """
    try:
        from gymnasium.envs.mujoco import mujoco_env as _gym_mujoco_env
    except Exception:  # pragma: no cover - gymnasium/mujoco not installed
        return 0

    assets_dir = Path(_gym_mujoco_env.__file__).parent / "assets"
    if not assets_dir.is_dir():
        return 0

    rewritten: dict[str, Path] = {}
    for xml_path in sorted(assets_dir.glob("*.xml")):
        try:
            text = xml_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(f'{a}="' in text for a in _LEGACY_MJCF_ATTRS):
            continue
        try:
            rewritten[xml_path.name] = _sanitized_asset(xml_path)
        except Exception:
            continue

    if not rewritten:
        return 0

    cls = getattr(_gym_mujoco_env, "MujocoEnv", None)
    if cls is None or getattr(cls, "_rltestbed_asset_patched", False):
        return len(rewritten)

    original_init = cls.__init__

    def _init(self, model_path, *args, **kwargs):
        # Only plain filenames are resolved against gymnasium's assets dir.
        if isinstance(model_path, str) and model_path in rewritten:
            model_path = str(rewritten[model_path])
        return original_init(self, model_path, *args, **kwargs)

    cls.__init__ = _init
    cls._rltestbed_asset_patched = True
    return len(rewritten)


def patch_mujoco_viewer_cleanup() -> bool:
    """Make Gymnasium's OffScreenViewer.free() tolerate a failed __init__.

    When an OpenGL context cannot be created, OffScreenViewer.__init__ raises
    before setting ``opengl_context``; its ``__del__`` then prints
    "AttributeError: 'OffScreenViewer' object has no attribute 'opengl_context'"
    for every attempt. The real error is reported by the caller; this only
    silences the misleading cleanup traceback.
    """
    import importlib.util

    if importlib.util.find_spec("mujoco") is None:
        return False
    try:
        from gymnasium.envs.mujoco import mujoco_rendering as mr
    except Exception:
        return False
    viewer = getattr(mr, "OffScreenViewer", None)
    if viewer is None or getattr(viewer.free, "_rltestbed_safe", False):
        return False
    original = viewer.free

    def free(self):
        if not hasattr(self, "opengl_context"):
            return
        original(self)

    free._rltestbed_safe = True
    viewer.free = free
    return True


def apply_compat_shims() -> dict:
    """Apply every compatibility shim exactly once. Returns a summary dict."""
    global _APPLIED
    if _APPLIED:
        return {"applied": False}

    patch_numpy_aliases()
    patched_assets = 0
    try:
        patched_assets = patch_mujoco_assets()
    except Exception:
        patched_assets = 0

    try:
        viewer_patched = patch_mujoco_viewer_cleanup()
    except Exception:
        viewer_patched = False

    _APPLIED = True
    return {"applied": True, "mujoco_assets_patched": patched_assets,
            "mujoco_viewer_cleanup_patched": viewer_patched}


def mujoco_available() -> bool:
    """True when MuJoCo + a compatible ``mujoco`` python package are importable."""
    try:
        import gymnasium as gym  # noqa: F401
        import mujoco  # noqa: F401
    except Exception:
        return False
    try:
        from gymnasium.envs.mujoco import mujoco_env  # noqa: F401
    except Exception:
        return False
    return True


def mujoco_version() -> str:
    try:
        import mujoco

        return getattr(mujoco, "__version__", "unknown")
    except Exception:
        return "not installed"


def cleanup_asset_cache() -> None:
    """Remove the temporary sanitized-asset cache (used by tests)."""
    if _XML_CACHE_DIR.exists():
        shutil.rmtree(_XML_CACHE_DIR, ignore_errors=True)


# Keep ``os`` importable for consumers that expect it here (it is used above).
__all__ = [
    "apply_compat_shims",
    "patch_numpy_aliases",
    "patch_mujoco_assets",
    "patch_mujoco_viewer_cleanup",
    "mujoco_available",
    "mujoco_version",
    "cleanup_asset_cache",
]
