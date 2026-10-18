"""Optional-dependency checks for environments.

An environment declares the extras it needs in ``EnvSpec.requires``. This
module answers "is that extra importable here?" without importing it (so a
broken native library cannot crash catalogue listing), and says how to fix it.
"""

import importlib.util
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# requirement key -> (import name, human label, install hint)
REQUIREMENTS: Dict[str, Tuple[str, str, str]] = {
    "box2d": ("Box2D", "Box2D", "pip install \"gymnasium[box2d]\""),
    "mujoco": ("mujoco", "MuJoCo", "pip install \"gymnasium[mujoco]\""),
    "minatar": ("minatar", "MinAtar", "pip install MinAtar"),
    "pettingzoo": ("pettingzoo", "PettingZoo", "pip install pettingzoo"),
    "rlcard": ("rlcard", "RLCard", "pip install rlcard"),
    "minigrid": ("minigrid", "MiniGrid", "pip install minigrid"),
    "torch": ("torch", "PyTorch", "pip install torch"),
}


@lru_cache(maxsize=None)
def is_installed(requirement: str) -> bool:
    module = REQUIREMENTS.get(requirement, (requirement,))[0]
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing(requires: List[str]) -> List[str]:
    return [r for r in requires if not is_installed(r)]


def unavailable_reason(requires: List[str]) -> Optional[str]:
    gone = missing(requires)
    if not gone:
        return None
    labels = [REQUIREMENTS.get(r, (r, r, ""))[1] for r in gone]
    hints = [REQUIREMENTS.get(r, (r, r, f"pip install {r}"))[2] for r in gone]
    return f"Requires {' + '.join(labels)}, which is not installed. Install with: {'; '.join(hints)}"


def installed_extras() -> Dict[str, bool]:
    return {key: is_installed(key) for key in REQUIREMENTS}
