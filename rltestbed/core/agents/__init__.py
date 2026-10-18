"""Agent registry: implemented classes, planned roadmap, specs and factories."""

from typing import Any, Dict, List

from .base import Agent, ParamSpec, build_spec
from .dqn import DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from .reinforce import ReinforceAgent, ReinforceBaselineAgent
from .tabular import QLearningAgent, SarsaAgent
from .actor_critic import A2CAgent
from .ppo import PPOAgent
from .random_agent import RandomAgent
from .continuous import DDPGAgent, TD3Agent, SACAgent
from .tabular_extra import (
    DoubleQLearningAgent,
    DynaQAgent,
    ExpectedSarsaAgent,
    MonteCarloControlAgent,
    ThompsonSamplingAgent,
    UCBAgent,
)
from .dqn_extra import C51Agent, PrioritizedDQNAgent
from .ppo_continuous import GaussianPPOAgent
from rltestbed.core.lm.agents import LM_AGENTS

AGENT_REGISTRY: Dict[str, type] = {
    "random": RandomAgent,
    "q_learning": QLearningAgent,
    "sarsa": SarsaAgent,
    "expected_sarsa": ExpectedSarsaAgent,
    "double_q": DoubleQLearningAgent,
    "mc_control": MonteCarloControlAgent,
    "dyna_q": DynaQAgent,
    # bandits / count-based exploration
    "ucb": UCBAgent,
    "thompson": ThompsonSamplingAgent,
    "dqn": DQNAgent,
    "dqn_double": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "dqn_per": PrioritizedDQNAgent,
    "c51": C51Agent,
    "reinforce": ReinforceAgent,
    "reinforce_baseline": ReinforceBaselineAgent,
    "a2c": A2CAgent,
    "ppo": PPOAgent,
    "ppo_continuous": GaussianPPOAgent,
    # language-model RL (trained by rltestbed.core.lm.trainer.SequenceTrainer)
    **LM_AGENTS,
    # continuous control (MuJoCo, Box action spaces)
    "ddpg": DDPGAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
}

# Honest capability matrix entries for methods that are NOT implemented yet.
# The UI shows these as 'planned' / 'experimental' - never as trainable.
PLANNED_ALGORITHMS: List[Dict[str, Any]] = [
    build_spec(
        name="noisy_dqn", label="NoisyNet DQN",
        description="Learned parametric noise in the network weights replaces "
        "epsilon-greedy exploration.",
        category="value-based", policy_type="off-policy", status="planned",
        status_reason="Needs a NoisyLinear layer and per-step noise resampling in the "
        "shared DQN loop; prioritised replay and C51 came first.",
        paper="Fortunato et al., 2018. Noisy Networks for Exploration. ICLR.",
    ),
    build_spec(
        name="rainbow", label="Rainbow",
        description="DQN with double-Q, dueling heads, prioritised replay, n-step "
        "returns, noisy nets and a distributional head, combined.",
        category="value-based", policy_type="off-policy", status="planned",
        status_reason="Composite of six components; four exist here individually "
        "(double, dueling, PER, C51). Waiting on NoisyNet and n-step returns.",
        paper="Hessel et al., 2018. Rainbow. AAAI.",
    ),
    build_spec(
        name="trpo", label="TRPO",
        description="Trust-region policy optimisation via a KL-constrained natural "
        "gradient step.",
        category="policy-gradient", policy_type="on-policy", status="planned",
        status_reason="Requires conjugate-gradient and line-search machinery. PPO "
        "approximates the same trust region with a clipped objective and is available.",
        paper="Schulman et al., 2015. Trust Region Policy Optimization. ICML.",
    ),
    build_spec(
        name="self_play_ppo", label="Self-play PPO",
        description="The opponent is a frozen snapshot of the learner, refreshed "
        "periodically, so difficulty rises with skill.",
        category="multi-agent", policy_type="on-policy", status="planned",
        multi_agent=True,
        status_reason="The game adapter currently plays a scripted random opponent. "
        "Self-play needs the trainer to hand a frozen policy copy to the environment.",
        paper="Silver et al., 2017; Bansal et al., 2018. Emergent complexity via multi-agent competition.",
    ),
    build_spec(
        name="independent_learners", label="Independent learners",
        description="One learner per seat, each treating the others as part of the "
        "environment. The simplest multi-agent baseline.",
        category="multi-agent", policy_type="mixed", status="planned", multi_agent=True,
        status_reason="Needs a multi-policy trainer; the training loop drives exactly "
        "one agent today.",
        paper="Tan, 1993. Multi-agent reinforcement learning: independent vs cooperative agents.",
    ),
    build_spec(
        name="bc", label="Behaviour cloning",
        description="Supervised learning of actions from a fixed dataset of "
        "demonstrations. The baseline for offline RL.",
        category="offline", policy_type="offline", status="planned",
        status_reason="Offline methods need a dataset store (record rollouts of a kept "
        "model, then train from them), which does not exist yet.",
        paper="Pomerleau, 1991; Torabi et al., 2018.",
    ),
    build_spec(
        name="cql", label="Conservative Q-learning",
        description="Offline Q-learning with a penalty that keeps values low on "
        "actions the dataset never took.",
        category="offline", policy_type="offline", status="planned",
        status_reason="Blocked on the same offline dataset pipeline as behaviour cloning.",
        paper="Kumar et al., 2020. Conservative Q-Learning for Offline RL. NeurIPS.",
    ),
    build_spec(
        name="rnd", label="Random network distillation",
        description="An intrinsic curiosity bonus: reward the agent for visiting "
        "states a predictor network cannot yet predict.",
        category="exploration", policy_type="mixed", status="planned",
        status_reason="Designed as a reward bonus wrapped around another learner; the "
        "trainer has no reward-wrapper hook yet.",
        paper="Burda et al., 2019. Exploration by Random Network Distillation. ICLR.",
    ),
]


def list_agents(include_planned: bool = True) -> List[Dict[str, Any]]:
    """All algorithm specs: implemented first (registry order), then planned."""
    out = [cls.SPEC for cls in AGENT_REGISTRY.values()]
    if include_planned:
        out += PLANNED_ALGORITHMS
    return out


def get_agent_class(name: str):
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Available: {sorted(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]


def default_params(name: str) -> Dict[str, Any]:
    return {p["name"]: p["default"] for p in get_agent_class(name).SPEC["params"]}


def create_agent(name: str, obs_dim: int, act_dim: int,
                 params: Dict[str, Any] | None = None, device: str = "cpu") -> Agent:
    params_full = default_params(name)
    if params:
        unknown = set(params) - set(params_full)
        if unknown:
            raise ValueError(f"Unknown hyperparameters for '{name}': {sorted(unknown)}")
        params_full.update(params)
    return get_agent_class(name)(
        obs_dim=obs_dim, act_dim=act_dim, params=params_full, device=device
    )
