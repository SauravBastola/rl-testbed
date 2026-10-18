"""Unified single-agent environment interface and rich spec metadata."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class EnvAdapter(Protocol):
    """Every environment in the testbed is normalized to this API:

    - reset() -> obs
    - step(action) -> (obs, reward, terminated, truncated, info)
    - obs_dim: flattened observation size (int)
    - act_dim: number of discrete actions (int)

    Optional extras an adapter may provide:
    - render(obs) -> str   ASCII/visual representation for the web preview
    """

    obs_dim: int
    act_dim: int

    def reset(self) -> Any: ...

    def step(self, action: int): ...

    def close(self): ...


@dataclass
class EnvSpec:
    """Static metadata about an environment - the single copy shown in UI/CLI."""

    name: str                      # unique key, e.g. "cartpole"
    label: str                     # display name
    category: str                  # builtin | gym | minatar | pettingzoo | probe | custom
    description: str = ""
    source: str = ""
    url: str = ""
    tags: List[str] = field(default_factory=list)
    max_steps: int = 1000
    difficulty: str = "medium"     # easy | medium | hard | experimental

    # capability metadata -------------------------------------------------
    obs_type: str = "vector"       # vector | onehot | image
    action_type: str = "discrete"  # discrete | continuous
    n_agents: int = 1
    action_masked: bool = False    # observations carry legal-action masks

    # teaching metadata ----------------------------------------------------
    reward_desc: str = ""
    termination_desc: str = ""
    typical_episode_length: int = 0  # 0 => default to max_steps
    recommended_agents: List[str] = field(default_factory=list)
    unsuitable_agents: List[str] = field(default_factory=list)
    required_wrappers: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    beginner_exercise: str = ""
    advanced_exercise: str = ""
    sparse_reward: bool = False
    # optional extras this env needs (keys of availability.REQUIREMENTS)
    requires: List[str] = field(default_factory=list)
    # explicit lab override; normally derived by rltestbed.labs.lab_for_env
    lab: str = ""
    # set when the dependency is installed but the env failed to construct
    probe_error: str = ""
    # return at which the task counts as solved (from Gymnasium's registry
    # where one is defined); drawn as a threshold line on learning curves
    reward_threshold: Optional[float] = None

    # lazily resolved by probing the env (may be None until first probe)
    obs_dim: Optional[int] = None
    act_dim: Optional[int] = None

    def __post_init__(self):
        if not self.typical_episode_length:
            self.typical_episode_length = self.max_steps

    def to_dict(self, resolve=False) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "tags": self.tags,
            "max_steps": self.max_steps,
            "difficulty": self.difficulty,
            "obs_type": self.obs_type,
            "action_type": self.action_type,
            "n_agents": self.n_agents,
            "action_masked": self.action_masked,
            "reward_desc": self.reward_desc,
            "termination_desc": self.termination_desc,
            "typical_episode_length": self.typical_episode_length,
            "recommended_agents": self.recommended_agents,
            "unsuitable_agents": self.unsuitable_agents,
            "required_wrappers": self.required_wrappers,
            "limitations": self.limitations,
            "beginner_exercise": self.beginner_exercise,
            "advanced_exercise": self.advanced_exercise,
            "sparse_reward": self.sparse_reward,
            "requires": self.requires,
            "reward_threshold": self.reward_threshold,
        }
        from .availability import unavailable_reason

        reason = unavailable_reason(self.requires) or (self.probe_error or None)
        d["available"] = reason is None
        d["unavailable_reason"] = reason
        from rltestbed.labs import lab_for_env

        d["lab"] = self.lab or lab_for_env(d)
        if resolve and reason is None:
            try:
                from rltestbed.core.envs import REGISTRY

                env = REGISTRY[self.name]["factory"]()
                d["obs_dim"], d["act_dim"] = env.obs_dim, env.act_dim
                env.close()
            except Exception as e:
                d["obs_dim"] = None
                d["act_dim"] = None
                d["probe_error"] = str(e)
        else:
            d["obs_dim"] = self.obs_dim
            d["act_dim"] = self.act_dim
        return d


# filled in by __init__.py
REGISTRY: Dict[str, Dict[str, Any]] = {}


def gym_reward_threshold(env_id: str) -> Optional[float]:
    """Solved threshold declared in Gymnasium's registry, without building the env."""
    try:
        from gymnasium.envs.registration import registry

        spec = registry.get(env_id)
        value = getattr(spec, "reward_threshold", None)
        return float(value) if value is not None else None
    except Exception:
        return None
