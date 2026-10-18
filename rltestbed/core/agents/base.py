"""Base agent interface, parameter specs and rich algorithm metadata.

Every agent exposes a SPEC dict consumed by the CLI and web app. The spec
carries three kinds of information:

1. Identity        - name, label, category, on/off-policy, status
2. Capabilities    - supported observation/action types, masking, multi-agent
3. Encyclopedia    - beginner/technical explanations, pseudocode, failure
                     modes, presets, recommended environments

The frontend renders this metadata directly; there is no duplicate copy of
algorithm documentation anywhere else.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ParamSpec:
    """Describes one hyperparameter so forms can be auto-generated."""

    name: str
    label: str
    default: Any
    type: str  # "float" | "int" | "bool" | "choice"
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: List[Any] = field(default_factory=list)
    advanced: bool = False  # hidden in beginner mode
    recommended: Optional[Any] = None  # suggested value shown in UI

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "default": self.default,
            "type": self.type,
            "help": self.help,
            "min": self.min,
            "max": self.max,
            "choices": self.choices,
            "advanced": self.advanced,
            "recommended": self.recommended if self.recommended is not None else self.default,
        }


def build_spec(
    name: str,
    label: str,
    description: str,
    *,
    category: str = "value-based",
    policy_type: str = "off-policy",
    status: str = "implemented",
    params: Optional[List[ParamSpec]] = None,
    presets: Optional[List[Dict[str, Any]]] = None,
    supports: Optional[Dict[str, Any]] = None,
    beginner: str = "",
    technical: str = "",
    intuition: str = "",
    update_rule: str = "",
    pseudocode: str = "",
    lifecycle: str = "",
    learns: str = "",
    exploration: str = "",
    replay_buffer: str = "Not used.",
    target_network: str = "Not used.",
    advantage: str = "Not used.",
    networks: str = "",
    failure_modes: Optional[List[str]] = None,
    obs_types: Optional[List[str]] = None,
    action_types: Optional[List[str]] = None,
    multi_agent: bool = False,
    recommended_envs: Optional[List[str]] = None,
    poor_envs: Optional[List[str]] = None,
    expected_metrics: Optional[List[Dict[str, str]]] = None,
    related: Optional[List[str]] = None,
    math: Optional[str] = None,
    status_reason: str = "",
    paper: str = "",
) -> Dict[str, Any]:
    """Construct a complete, UI-ready agent spec with sensible defaults.

    ``math`` is Markdown with ``$inline$`` LaTeX. When omitted it is looked up
    in :mod:`.theory`, so the theory table never has to be wired by hand.
    ``status_reason`` explains *why* a non-implemented method is not trainable;
    the UI shows it verbatim instead of a generic "planned" label.
    """
    if math is None:
        from .theory import THEORY

        math = THEORY.get(name, "")
    return {
        "name": name,
        "label": label,
        "description": description,
        "category": category,
        "type": policy_type,
        "status": status,  # implemented | experimental | planned | unsupported
        "status_reason": status_reason,
        "paper": paper,
        "params": [p.to_dict() for p in (params or [])],
        "presets": presets or [],
        "supports": supports or {
            "obs_types": obs_types or ["vector", "onehot"],
            "action_types": action_types or ["discrete"],
            "multi_agent": multi_agent,
            "action_mask": True,
            "recurrent": False,
        },
        # encyclopedia ------------------------------------------------------
        "docs": {
            "beginner": beginner,
            "technical": technical,
            "intuition": intuition,
            "update_rule": update_rule,
            "pseudocode": pseudocode,
            "lifecycle": lifecycle,
            "learns": learns,
            "exploration": exploration,
            "replay_buffer": replay_buffer,
            "target_network": target_network,
            "advantage": advantage,
            "networks": networks,
            "failure_modes": failure_modes or [],
            "recommended_envs": recommended_envs or [],
            "poor_envs": poor_envs or [],
            "expected_metrics": expected_metrics or [],
            "related": related or [],
            "math": math,
        },
    }


class Agent(ABC):
    """A learning agent interacting via a normalized single-agent env API.

    Conventions:
    - Observations may be raw arrays/vectors OR dicts {"obs": ..., "action_mask": [...]}
      (action masks come from game-like envs).
    - `get_action(obs, valid_actions=None)` returns an int action.
    - Off-policy agents implement update(obs, action, reward, next_obs, done) per step.
    - On-policy agents buffer internally and finish the episode with end_of_episode().
    """

    #: subclasses set this to a dict describing the agent for the UI
    SPEC: Dict[str, Any] = {}

    TRAIN: bool = True

    def __init__(self, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        self.act_dim = int(act_dim)
        self.params = dict(params)
        self.device = device

    # ------------------------------------------------------------------ #
    def _unpack(self, obs):
        """Return (state_array_or_value, action_mask or None)."""
        mask = None
        if isinstance(obs, dict):
            mask = obs.get("action_mask")
            obs = obs.get("obs", list(obs.values())[0])
        return obs, mask

    @abstractmethod
    def get_action(self, obs, valid_actions=None) -> int: ...

    @abstractmethod
    def update(self, obs, action, reward, next_obs, done): ...

    def end_of_episode(self):
        """Called once at episode end (on-policy agents use this to learn)."""

    def set_training_progress(self, progress: float):
        """Optional hook: called with overall training progress in [0, 1].
        Agents may use this for lr / exploration annealing."""

    def set_action_bounds(self, low, high):
        """Optional hook: informs the agent of the environment's action limits.

        Only continuous-control agents care about this - they emit actions in
        a normalised range and rescale to these bounds. The default is a
        no-op so discrete agents need not implement it.
        """
        self.act_low = np.asarray(low, dtype=np.float32).flatten()
        self.act_high = np.asarray(high, dtype=np.float32).flatten()

    def finalize_training(self):
        """Flush any pending on-policy rollout before a checkpoint is written."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Scalar training metrics for live charts."""

    @abstractmethod
    def save_model(self, path): ...

    @abstractmethod
    def load_model(self, path): ...

    # helpers -----------------------------------------------------------
    def _mask_or_range(self, mask):
        if mask:
            return [i for i, v in enumerate(mask) if v]
        return list(range(self.act_dim))

    def describe(self) -> str:
        return f"{type(self).__name__}(act_dim={self.act_dim})"


def rolling(x, maxlen=500):
    d = deque(maxlen=maxlen)
    d.extend(x)
    return d
