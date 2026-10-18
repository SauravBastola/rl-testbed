"""Diagnostic benchmark catalog.

Each benchmark pairs a probe environment with an explanation of what it
verifies, what failure means, and which algorithm families should pass.
Historical results are derived at request time from runs on the linked probe.
"""

from typing import Any, Dict, List

BENCHMARKS: List[Dict[str, Any]] = [
    dict(
        id="sanity",
        probe="probe1", pass_threshold=0.99, episodes=50,
        title="Policy sanity",
        family="basic",
        tests="Can the implementation learn *anything*? One state, one action, constant +1 reward.",
        why="Catches broken optimisers, wrong loss signs, and pipeline bugs before you waste GPU-hours.",
        expected="Reward reaches 1.0 within a few dozen episodes for any working learner.",
        failure_suggests="A plumbing bug: gradients not applied, agent never updated, or env/agent mismatch.",
        should_pass=["dqn", "dqn_double", "dueling_dqn", "dqn_per", "c51", "reinforce", "reinforce_baseline", "a2c", "ppo", "q_learning", "sarsa", "expected_sarsa", "double_q", "mc_control", "dyna_q", "ucb", "thompson"],
        how_to="Train any algorithm on probe1 for 30-50 episodes; every eval should equal 1.0.",
    ),
    dict(
        id="state-conditioning",
        probe="probe2", pass_threshold=0.95, episodes=200,
        title="State conditioning",
        family="basic",
        tests="Does the policy depend on observations at all (action 1 always beats action 0)?",
        why="Detects blind policies - e.g. networks that ignore inputs or agents acting from stale buffers.",
        expected="Greedy eval converges to 1.0 as the agent reliably picks action 1.",
        failure_suggests="Observations are not reaching the network, or exploration noise dominates forever.",
        should_pass=["dqn", "dqn_double", "dueling_dqn", "dqn_per", "c51", "reinforce_baseline", "a2c", "ppo", "ucb", "thompson"],
        how_to="Train DQN on probe2, 50 episodes, then confirm greedy eval hits 1.0.",
    ),
    dict(
        id="delayed-reward",
        probe="probe3", pass_threshold=0.95, episodes=300,
        title="Delayed reward & credit assignment",
        family="credit",
        tests="Two-step chain where only the second action pays out (+1). Requires correct discount propagation.",
        why="The essence of RL is propagating future reward to earlier decisions. This verifies bootstrapping/discounting.",
        expected="Agent learns action sequence [0 then 1]; eval reward → 1.0.",
        failure_suggests="gamma too low, targets computed without next-state values, or returns misaligned by an offset.",
        should_pass=["dqn", "dqn_double", "dqn_per", "q_learning", "expected_sarsa", "double_q", "dyna_q", "mc_control", "ppo", "reinforce_baseline"],
        how_to="Train DQN (γ=0.9+) or Q-learning on probe3 for 100-200 episodes.",
    ),
    dict(
        id="stochastic-reward",
        probe="probe4", pass_threshold=0.6, episodes=300,
        title="Stochastic rewards",
        family="robustness",
        tests="Same state/action pair yields different rewards (p=0.8 vs p=0.2 arms).",
        why="Real environments are noisy. This checks value estimates converge through variance instead of oscillating.",
        expected="Slower but steady convergence toward preferring action 1; loss stays bounded.",
        failure_suggests="Learning rate too high for the noise level, or no averaging in updates.",
        should_pass=["dqn", "dqn_double", "dqn_per", "ppo", "ucb", "thompson"],
        how_to="Train DQN 200 episodes on probe4; mean reward should exceed 0.6.",
    ),
    dict(
        id="exploration",
        probe="probe5", pass_threshold=0.6, episodes=300,
        title="Exploration vs exploitation",
        family="exploration",
        tests="Classic two-armed bandit (arms pay 0.8 / 0.2). Pure exploitation of early luck fails.",
        why="Verifies the exploration mechanism samples enough to find the better arm.",
        expected="With healthy epsilon decay the agent locks onto arm 1; eval ≈ 0.8.",
        failure_suggests="Epsilon decayed too fast, or entropy bonus too weak.",
        should_pass=["dqn", "dqn_double", "q_learning", "ppo", "ucb", "thompson", "double_q"],
        how_to="Train DQN 300 episodes on probe5 with default epsilon schedule.",
    ),
    dict(
        id="continuous",
        probe="probe6", pass_threshold=0.9, episodes=300,
        title="Continuous control sanity",
        family="continuous",
        tests="One step, one continuous action in [-1, 1]; reward 1 - |a - 0.5|.",
        why="The only probe continuous-action learners can run. Checks action "
        "scaling, bound handling and that the policy moves toward higher reward.",
        expected="The greedy action settles within 0.1 of 0.5, so evaluation reward exceeds 0.9.",
        failure_suggests="Actions not rescaled to the bounds, a squashing bug, or "
        "exploration noise left on during evaluation.",
        should_pass=["sac", "td3", "ddpg", "ppo_continuous"],
        how_to="Train SAC on probe6 for 300 episodes; evaluation should exceed 0.9.",
    ),
]


# Probe episodes last one or two steps, so a few hundred episodes are only a few
# hundred transitions. Defaults tuned for CartPole-length episodes (a 2048-step
# PPO rollout, a target sync every 200 updates) would barely learn at all in
# that budget, and a probe failure would point at the budget rather than a bug.
# Diagnostic runs therefore shrink these intervals. Keys are matched against
# each algorithm's own ParamSpec, so unknown names are simply ignored.
DIAGNOSTIC_OVERRIDES: Dict[str, Any] = {
    "target_update_freq": 20,
    "rollout_steps": 64,
    "minibatch_size": 32,
    # continuous learners: start updating after a few dozen one-step episodes
    "warmup_steps": 64,
    "batch_size": 32,
}


def diagnostic_params(agent_spec: Dict[str, Any]) -> Dict[str, Any]:
    names = {p["name"] for p in agent_spec.get("params", [])}
    return {k: v for k, v in DIAGNOSTIC_OVERRIDES.items() if k in names}


def benchmark_dict() -> List[Dict[str, Any]]:
    return BENCHMARKS


def verdict(benchmark: Dict[str, Any], run: Dict[str, Any] | None) -> str:
    """pass | fail | running | untested for one run on a benchmark's probe."""
    if not run:
        return "untested"
    if run.get("status") in ("queued", "created", "running", "stopping"):
        return "running"
    best = run.get("best_eval_reward")
    if best is not None and best >= benchmark["pass_threshold"]:
        return "pass"
    if run.get("status") == "completed":
        return "fail"
    return "untested"
