"""Environment catalogues for the larger Gymnasium suites.

These are kept in their own module because they are large, mostly-static
metadata tables.  Each catalogue is a list of plain dicts consumed by
:func:`rltestbed.core.envs.build_suite_specs`, which turns them into
``EnvSpec`` objects and registers them.

Two suites are represented here:

``MUJOCO_ENVS``
    Continuous-control locomotion and manipulation built on the MuJoCo
    physics engine.  All of them have continuous action spaces, so they are
    only trainable with the continuous-control algorithms (DDPG / TD3 / SAC).

``TOY_TEXT_ENVS``
    Small tabular MDPs with discrete observation spaces.  These are the
    textbook environments for tabular methods (Q-learning, SARSA).

``CONTINUOUS_ENVS``
    Classic-control tasks with continuous actions that do not require the
    MuJoCo dependency but still exercise continuous-control algorithms.
"""

from typing import Any, Callable, Dict, List

# --------------------------------------------------------------------- #
# MuJoCo - continuous control
# --------------------------------------------------------------------- #
# Fields mirror the dicts in rltestbed/core/envs/__init__.py.
# `act_range` documents the Box bounds; it is informational only.
MUJOCO_ENVS: List[Dict[str, Any]] = [
    # ---- CartPoles -------------------------------------------------- #
    dict(
        name="inverted_pendulum", env_id="InvertedPendulum-v4",
        description=(
            "MuJoCo's continuous-force version of CartPole. A cart slides along a "
            "rail and you apply a horizontal force to keep the pole upright. The "
            "gentlest possible introduction to continuous control."
        ),
        tags=["mujoco", "continuous", "balance"], max_steps=1000, difficulty="easy",
        obs_type="vector", action_type="continuous", act_range=(-3.0, 3.0),
        group="CartPoles",
        reward_desc="+1 alive bonus for every step the pole stays within ~0.2 rad "
        "of vertical and the cart stays on the rail.",
        termination_desc="Pole angle leaves the healthy range, the cart slides off "
        "the rail, or 1000 steps elapse.",
        typical_episode_length=1000,
        recommended_agents=["ddpg", "td3", "sac", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Train SAC for ~100 episodes. This is usually the fastest "
        "MuJoCo task to show clear improvement, so it's the right first check that "
        "your continuous-control pipeline works.",
        advanced_exercise="Compare SAC vs TD3 vs DDPG on sample efficiency. DDPG's "
        "deterministic policy should plateau earlier and be far more seed-sensitive.",
        limitations=["Action is a single continuous force - almost too easy; it will "
                     "not discriminate between good and mediocre implementations."],
    ),
    dict(
        name="inverted_double_pendulum", env_id="InvertedDoublePendulum-v4",
        description=(
            "Two-link pendulum balanced on a sliding cart. A single continuous "
            "force must control a much less stable system than InvertedPendulum - "
            "the second link makes the dynamics genuinely harder."
        ),
        tags=["mujoco", "continuous", "balance"], max_steps=1000, difficulty="medium",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="CartPoles",
        reward_desc="Alive bonus plus quadratic penalties on the pole angles and "
        "their velocities; the agent is pushed toward a tight, still equilibrium.",
        termination_desc="Either link tilts too far, the cart leaves the rail, or "
        "1000 steps elapse.",
        typical_episode_length=1000,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Run SAC for a few hundred episodes. Unlike the single "
        "pendulum, expect a long flat period before the balancing behaviour locks in.",
        advanced_exercise="Sweep the action-smoothing / policy-delay settings of TD3 "
        "and see how much of the gain comes from delayed policy updates.",
        limitations=["Needs meaningfully more samples than InvertedPendulum."],
    ),
    # ---- Arms -------------------------------------------------------- #
    dict(
        name="reacher", env_id="Reacher-v4",
        description=(
            "A two-link planar arm must touch a randomly placed target. Short, "
            "well-shaped episodes make it a friendly manipulation task."
        ),
        tags=["mujoco", "continuous", "manipulation"], max_steps=50, difficulty="easy",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="Arms",
        reward_desc="Negative distance from fingertip to target, minus a quadratic "
        "control cost.",
        termination_desc="Fixed 50-step horizon - there is no early termination.",
        typical_episode_length=50,
        recommended_agents=["sac", "td3", "ddpg", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa"],
        beginner_exercise="SAC converges here in a few hundred episodes. Use it to "
        "validate your setup before moving to locomotion.",
        advanced_exercise="Because episodes are fixed-length and dense, Reacher is a "
        "good place to study how target entropy (SAC's alpha) changes exploration.",
        limitations=["Short horizon and dense reward mean it rewards tuning more "
                     "than genuine long-horizon reasoning."],
    ),
    dict(
        name="pusher", env_id="Pusher-v4",
        description=(
            "A 7-DOF arm pushes a cylinder across a table toward a target site. "
            "Harder than Reacher because the reward is a sum of three competing "
            "distances (hand-to-object, object-to-target, control cost)."
        ),
        tags=["mujoco", "continuous", "manipulation"], max_steps=100, difficulty="hard",
        obs_type="vector", action_type="continuous", act_range=(-2.0, 2.0),
        group="Arms",
        reward_desc="-(distance fingertip to object) - (distance object to target) "
        "- control cost.",
        termination_desc="Fixed 100-step horizon.",
        typical_episode_length=100,
        recommended_agents=["sac", "td3"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Start here only after Reacher works. Watch whether the "
        "agent learns to reach the object before it learns to push it anywhere useful.",
        advanced_exercise="This is a classic reward-shaping study: try adding a "
        "staged curriculum and compare time-to-solve.",
        limitations=["Sparse-ish composite reward; often needs 1M+ steps and careful "
                     "entropy tuning."],
    ),
    # ---- 2D runners --------------------------------------------------- #
    dict(
        name="half_cheetah", env_id="HalfCheetah-v4",
        description=(
            "A 2-dimensional 'cheetah' with a fixed torso and two legs (6 joints) "
            "must learn to run forward as fast as possible. The standard "
            "sample-efficiency benchmark for continuous control."
        ),
        tags=["mujoco", "continuous", "locomotion"], max_steps=1000, difficulty="medium",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="2D Runners",
        reward_desc="Forward velocity reward minus a quadratic control cost.",
        termination_desc="Fixed 1000-step horizon (no 'unhealthy' termination, which "
        "is why it's more forgiving than Hopper/Walker2d).",
        typical_episode_length=1000,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="SAC for ~300 episodes should produce visibly faster "
        "running. Compare against DDPG to feel the sample-efficiency gap.",
        advanced_exercise="Benchmark SAC vs TD3 vs PPO at a fixed step budget (e.g. "
        "1M) and report mean return over 5 seeds - single-seed MuJoCo numbers are "
        "notoriously misleading.",
        limitations=["Sensitive to reward scale and to the control-cost weight; "
                     "results are not comparable across gymnasium versions."],
    ),
    dict(
        name="hopper", env_id="Hopper-v4",
        description=(
            "A 2D one-legged robot (3 joints) must hop forward without falling. "
            "Adds a 'stay healthy' constraint that HalfCheetah lacks, so a bad "
            "policy terminates early instead of just scoring badly."
        ),
        tags=["mujoco", "continuous", "locomotion"], max_steps=1000, difficulty="hard",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="2D Runners",
        reward_desc="Forward velocity - control cost + a +1 alive bonus while the "
        "torso height and joint angles stay in the healthy range.",
        termination_desc="Unhealthy state (torso too low/high, joint out of range, "
        "or non-finite observation) or 1000 steps.",
        typical_episode_length=600,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Expect early collapse: the agent learns to fall slowly "
        "before it learns to hop. Use the alive bonus to read progress.",
        advanced_exercise="Study premature termination: compare TD3 with and without "
        "target-policy smoothing on fall rate.",
        limitations=["Early termination makes returns high-variance across seeds."],
    ),
    dict(
        name="walker2d", env_id="Walker2d-v4",
        description=(
            "A 2D bipedal walker (6 joints) must stay upright and move forward. "
            "Like Hopper it terminates on unhealthy states, but with twice the "
            "degrees of freedom."
        ),
        tags=["mujoco", "continuous", "locomotion"], max_steps=1000, difficulty="hard",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="2D Runners",
        reward_desc="Forward velocity - control cost + alive bonus while healthy.",
        termination_desc="Unhealthy state (torso angle/height out of range) or 1000 "
        "steps.",
        typical_episode_length=700,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Give SAC a few hundred episodes. Watch the episode length "
        "rise before the return does - staying alive is learned first.",
        advanced_exercise="Compare Walker2d against Hopper: which is more sensitive "
        "to the number of critic target updates?",
        limitations=["Slower and higher-variance than HalfCheetah; needs multiple "
                     "seeds to draw conclusions."],
    ),
    # ---- Swimmers ------------------------------------------------------ #
    dict(
        name="swimmer", env_id="Swimmer-v4",
        description=(
            "A 3-link snake-like robot swims forward in a viscous fluid. Famous "
            "as an exploration trap: the gradient signal is weak and it is easy "
            "to converge to a local optimum where the robot just wriggles in place."
        ),
        tags=["mujoco", "continuous", "locomotion"], max_steps=1000, difficulty="hard",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="Swimmers",
        reward_desc="Forward velocity minus a quadratic control cost.",
        termination_desc="Fixed 1000-step horizon.",
        typical_episode_length=1000,
        recommended_agents=["sac", "td3"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Use this as an exploration stress test. If your agent "
        "stagnates, that is the expected result, not necessarily a bug.",
        advanced_exercise="Compare SAC's automatic entropy tuning against a fixed "
        "alpha - Swimmer is where entropy tuning matters most.",
        limitations=["Weak, deceptive gradients; single-seed results are meaningless.",
                     "Requires the MuJoCo compat shim (rltestbed.compat) on the "
                     "gymnasium version pinned by this project."],
    ),
    # ---- Quadruped ------------------------------------------------------ #
    dict(
        name="ant", env_id="Ant-v4",
        description=(
            "A 3D quadruped with 8 joints must learn to walk/run forward. Contact "
            "dynamics make it substantially richer than the 2D runners."
        ),
        tags=["mujoco", "continuous", "locomotion", "3d"], max_steps=1000,
        difficulty="hard",
        obs_type="vector", action_type="continuous", act_range=(-1.0, 1.0),
        group="Quadruped",
        reward_desc="Forward velocity - control cost - contact cost + alive bonus "
        "while the torso stays healthy.",
        termination_desc="Unhealthy state (torso height/angle out of range, "
        "non-finite observation) or 1000 steps.",
        typical_episode_length=800,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Budget several hundred episodes and evaluate greedily - "
        "the training-time return can look good while the greedy policy falls over.",
        advanced_exercise="Ant is the standard place to test whether an algorithm "
        "scales to higher-dimensional control. Compare at a fixed 3M-step budget.",
        limitations=["Needs 1-3M steps and multiple seeds; expensive on CPU."],
    ),
    # ---- Humanoids ------------------------------------------------------ #
    dict(
        name="humanoid", env_id="Humanoid-v4",
        description=(
            "A 3D humanoid with 17 joints must learn to walk forward. The "
            "highest-dimensional control task in the suite (376-dim observations) "
            "and a genuine test of algorithm scalability."
        ),
        tags=["mujoco", "continuous", "locomotion", "3d"], max_steps=1000,
        difficulty="expert",
        obs_type="vector", action_type="continuous", act_range=(-0.4, 0.4),
        group="Humanoids",
        reward_desc="Forward velocity - control cost - impact cost + alive bonus "
        "while the torso remains healthy.",
        termination_desc="Unhealthy state (torso height out of range, non-finite "
        "observation) or 1000 steps.",
        typical_episode_length=700,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Only attempt this after Ant works. Treat anything above a "
        "random baseline in the first million steps as progress.",
        advanced_exercise="Humanoid is where on-policy methods with many parallel "
        "environments usually win; measure wall-clock, not just sample efficiency.",
        limitations=["Very expensive: expect millions of steps and long wall-clock "
                     "times on CPU. Not suitable as a first MuJoCo task."],
    ),
    dict(
        name="humanoid_standup", env_id="HumanoidStandup-v4",
        description=(
            "The 3D humanoid starts lying on the ground and must stand up. "
            "Reward shaping is dense, so it is more tractable than Humanoid, but "
            "the state space is just as large."
        ),
        tags=["mujoco", "continuous", "locomotion", "3d"], max_steps=1000,
        difficulty="expert",
        obs_type="vector", action_type="continuous", act_range=(-0.4, 0.4),
        group="Humanoids",
        reward_desc="Shaped reward for raising the torso, minus control and impact "
        "costs.",
        termination_desc="Fixed 1000-step horizon (no unhealthy termination).",
        typical_episode_length=1000,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa", "random"],
        beginner_exercise="Prefer this over Humanoid if you want a 17-DOF task that "
        "actually shows learning progress in reasonable time.",
        advanced_exercise="Compare standing-up behaviour across SAC entropy targets; "
        "the task has many valid solutions.",
        limitations=["376-dim observations; expensive to train on CPU."],
    ),
]

MUJOCO_URLS = {
    "inverted_pendulum": "https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/",
    "inverted_double_pendulum": "https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/",
    "reacher": "https://gymnasium.farama.org/environments/mujoco/reacher/",
    "pusher": "https://gymnasium.farama.org/environments/mujoco/pusher/",
    "half_cheetah": "https://gymnasium.farama.org/environments/mujoco/half_cheetah/",
    "hopper": "https://gymnasium.farama.org/environments/mujoco/hopper/",
    "walker2d": "https://gymnasium.farama.org/environments/mujoco/walker2d/",
    "swimmer": "https://gymnasium.farama.org/environments/mujoco/swimmer/",
    "ant": "https://gymnasium.farama.org/environments/mujoco/ant/",
    "humanoid": "https://gymnasium.farama.org/environments/mujoco/humanoid/",
    "humanoid_standup": "https://gymnasium.farama.org/environments/mujoco/humanoidstandup/",
}

# --------------------------------------------------------------------- #
# Toy text - small tabular MDPs
# --------------------------------------------------------------------- #
TOY_TEXT_ENVS: List[Dict[str, Any]] = [
    dict(
        name="frozen_lake", env_id="FrozenLake-v1",
        description=(
            "Cross a frozen lake from start to goal without falling through a hole. "
            "The ice is slippery, so each move only sometimes goes where you "
            "intended - the canonical demonstration of why stochastic environments "
            "need more than greedy value iteration."
        ),
        tags=["toy-text", "grid", "tabular", "stochastic"], max_steps=100,
        difficulty="easy", obs_type="onehot", group="Toy Text",
        reward_desc="+1 for reaching the goal, 0 otherwise.",
        termination_desc="Reaching the goal, falling into a hole, or 100 steps.",
        typical_episode_length=20,
        recommended_agents=["q_learning", "sarsa", "dqn"],
        unsuitable_agents=["random"],
        beginner_exercise="Run Q-learning for a few thousand episodes in the "
        "slippery setting. Note that optimistic exploration matters: with too little "
        "exploration the agent never finds the goal.",
        advanced_exercise="Compare Q-learning against SARSA: SARSA learns the safer "
        "path that hugs the edges of the holes, Q-learning learns the shorter but "
        "riskier one. This is the cleanest demonstration of the on/off-policy "
        "difference in the whole catalogue.",
        limitations=["Tiny state space (16 states) - deep nets are overkill; this is "
                     "a tabular-methods environment."],
    ),
    dict(
        name="cliff_walking", env_id="CliffWalking-v0",
        description=(
            "Walk from start to goal along a cliff edge. The shortest path hugs the "
            "cliff and is dangerous; the safe path is much longer. Classic "
            "Sutton & Barto example contrasting SARSA and Q-learning."
        ),
        tags=["toy-text", "grid", "tabular"], max_steps=200, difficulty="easy",
        obs_type="onehot", group="Toy Text",
        reward_desc="-1 per step; -100 for stepping off the cliff (which also sends "
        "the agent back to the start).",
        termination_desc="Reaching the goal. Falling off resets to the start rather "
        "than ending the episode.",
        typical_episode_length=17,
        recommended_agents=["sarsa", "q_learning"],
        unsuitable_agents=["random"],
        beginner_exercise="Train SARSA and Q-learning side by side. SARSA's final "
        "return is worse but its online return is better - it avoids the cliff.",
        advanced_exercise="Plot online return (not final policy) for both algorithms. "
        "The point of the example is that the 'worse' policy is the safer learner.",
        limitations=["48 states; purely tabular."],
    ),
    dict(
        name="taxi", env_id="Taxi-v3",
        description=(
            "Drive a taxi: pick up a passenger at one of four locations and drop "
            "them at another, navigating walls in a 5x5 grid. The first "
            "environment where the state is genuinely composed of several facts "
            "(taxi position, passenger location, destination)."
        ),
        tags=["toy-text", "grid", "tabular"], max_steps=200, difficulty="medium",
        obs_type="onehot", group="Toy Text",
        reward_desc="+20 for a successful drop-off, -1 per step, -10 for an illegal "
        "pick-up or drop-off.",
        termination_desc="Successful drop-off or 200 steps.",
        typical_episode_length=15,
        recommended_agents=["q_learning", "sarsa", "dqn"],
        unsuitable_agents=["random"],
        beginner_exercise="Q-learning over ~5-10k episodes. 500 discrete states is "
        "still tabular but large enough that learning takes real time.",
        advanced_exercise="Because illegal moves are penalised, this is a good place "
        "to study epsilon-greedy vs softmax exploration.",
        limitations=["500 states - still tabular; a neural net will not help."],
    ),
    dict(
        name="blackjack", env_id="Blackjack-v1",
        description=(
            "Play Blackjack against a fixed dealer policy. The observation is a "
            "tuple (player sum, dealer showing card, usable ace) rather than a "
            "vector, which makes it a useful test of observation handling."
        ),
        tags=["toy-text", "card-game", "tabular"], max_steps=1, difficulty="medium",
        obs_type="onehot", group="Toy Text",
        reward_desc="+1 win, -1 loss, 0 draw (with the optional natural bonus).",
        termination_desc="Single decision per episode: stick or hit until bust.",
        typical_episode_length=1,
        recommended_agents=["q_learning", "sarsa"],
        unsuitable_agents=["dqn", "ppo", "a2c"],
        beginner_exercise="Monte-Carlo control shines here because episodes terminate "
        "and there is no bootstrapping bias. Compare against Q-learning.",
        advanced_exercise="Compare first-visit Monte-Carlo against TD(0) on the same "
        "data - a clean illustration of bias vs variance in RL.",
        limitations=["Tuple observation space (not a flat vector); the adapter "
                     "one-hot encodes and concatenates the components. Episodes are "
                     "one decision long, so it is not a sequential-credit task."],
    ),
]

TOY_TEXT_URLS = {
    "frozen_lake": "https://gymnasium.farama.org/environments/toy_text/frozen_lake/",
    "cliff_walking": "https://gymnasium.farama.org/environments/toy_text/cliff_walking/",
    "taxi": "https://gymnasium.farama.org/environments/toy_text/taxi/",
    "blackjack": "https://gymnasium.farama.org/environments/toy_text/blackjack/",
}

# --------------------------------------------------------------------- #
# Continuous classics (no MuJoCo dependency)
# --------------------------------------------------------------------- #
CONTINUOUS_ENVS: List[Dict[str, Any]] = [
    dict(
        name="mountain_car_continuous", env_id="MountainCarContinuous-v0",
        description=(
            "The continuous-torque version of Mountain Car. Instead of three "
            "discrete thrusts you apply a real-valued force in [-1, 1], and the "
            "magnitude of that force is penalised - so the agent must learn to be "
            "both effective and efficient."
        ),
        tags=["classic-control", "continuous", "sparse-reward"], max_steps=999,
        difficulty="medium", obs_type="vector", action_type="continuous",
        act_range=(-1.0, 1.0), group="Classic Control",
        reward_desc="+100 for reaching the goal, minus the sum of squared actions "
        "over the episode. Step reward is 0 otherwise.",
        termination_desc="Reaching the goal flag or 999 steps.",
        typical_episode_length=999,
        recommended_agents=["sac", "td3", "ddpg", "ppo"],
        unsuitable_agents=["dqn", "q_learning", "sarsa"],
        beginner_exercise="Because the action is penalised, a sloppy policy that "
        "thrashes the engine scores worse than a gentle one. Watch the action "
        "magnitude, not just the return.",
        advanced_exercise="Compare DDPG against SAC here: DDPG's deterministic "
        "policy tends to get stuck in the valley, SAC's entropy bonus keeps "
        "exploring the momentum strategy.",
        limitations=["Sparse reward: the shaping signal only appears once the car "
                     "reaches the flag, so early training looks completely flat."],
    ),
]

CONTINUOUS_URLS = {
    "mountain_car_continuous": "https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/",
}


def _tags_for(entry: Dict[str, Any]) -> List[str]:
    """Merge the declared tags with the suite subgroup (e.g. '2d runners')."""
    tags = [t for t in entry.get("tags", []) if t]
    group = (entry.get("group") or "").strip().lower()
    if group and group not in tags:
        tags.append(group)
    return tags


def build_suite_specs(
    env_class,
    registry: Dict[str, Dict[str, Any]],
    include_mujoco: bool = True,
) -> List[str]:
    """Create ``EnvSpec`` objects from the catalogues and register them.

    Returns the list of environment names that were successfully registered.
    MuJoCo environments are probed first: if the dependency is missing or the
    engine refuses to initialise, the whole MuJoCo suite is skipped rather
    than crashing the application at import time.
    """
    from .adapters import GymAdapter
    from .base import gym_reward_threshold

    registered: List[str] = []

    suites = [
        ("toy-text", TOY_TEXT_ENVS, TOY_TEXT_URLS, True),
        ("continuous", CONTINUOUS_ENVS, CONTINUOUS_URLS, True),
        ("mujoco", MUJOCO_ENVS, MUJOCO_URLS, include_mujoco),
    ]

    from .availability import is_installed

    for category, entries, urls, enabled in suites:
        if not enabled:
            continue
        requires = ["mujoco"] if category == "mujoco" else []
        installed = all(is_installed(r) for r in requires)
        suite_broken = ""
        for entry in entries:
            env_id = entry["env_id"]
            name = entry["name"]
            # Probe for real dimensions. A missing or broken dependency must
            # never take the application down: the env is still registered so
            # the catalogue can say *why* it cannot be used.
            obs_dim = act_dim = None
            probe_error = ""
            if installed and not suite_broken:
                try:
                    probe = GymAdapter(env_id)
                    obs_dim, act_dim = probe.obs_dim, probe.act_dim
                    probe.close()
                except Exception as exc:  # pragma: no cover - depends on host
                    probe_error = f"{env_id} failed to start: {type(exc).__name__}: {exc}"
                    if category == "mujoco":
                        suite_broken = probe_error
            elif suite_broken:
                probe_error = suite_broken

            spec = env_class(
                name=name,
                label=entry.get("label", env_id),
                category=category,
                description=entry["description"],
                source="gymnasium",
                url=urls.get(name, ""),
                tags=_tags_for(entry),
                max_steps=entry["max_steps"],
                difficulty=entry["difficulty"],
                obs_type=entry["obs_type"],
                action_type=entry.get("action_type", "discrete"),
                reward_desc=entry.get("reward_desc", ""),
                termination_desc=entry.get("termination_desc", ""),
                typical_episode_length=entry.get(
                    "typical_episode_length", entry["max_steps"]
                ),
                recommended_agents=list(entry.get("recommended_agents", [])),
                unsuitable_agents=list(entry.get("unsuitable_agents", [])),
                limitations=list(entry.get("limitations", [])),
                beginner_exercise=entry.get("beginner_exercise", ""),
                advanced_exercise=entry.get("advanced_exercise", ""),
                sparse_reward="sparse" in " ".join(entry.get("tags", [])),
                requires=requires,
                probe_error=probe_error,
                reward_threshold=gym_reward_threshold(env_id),
                obs_dim=obs_dim,
                act_dim=act_dim,
            )
            registry[name] = {"spec": spec, "factory": _factory_for(env_id)}
            registered.append(name)

    return registered


def _factory_for(env_id: str) -> Callable[[], Any]:
    """Return a zero-argument factory that builds a fresh adapter for `env_id`."""

    def _make():
        from .adapters import GymAdapter

        return GymAdapter(env_id)

    return _make
