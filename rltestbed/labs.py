"""Labs: workspaces that group the catalogue by the *shape of the learning problem*.

A lab is pure data. It owns a slice of the environment catalogue, an ordering
of algorithms that suit that slice, the metrics worth looking at first, and a
short curriculum. Every surface (API, CLI, web UI) reads labs from here; adding
a lab is a matter of appending to ``LABS``.

Membership is decided by :func:`lab_for_env`. A run belongs to the lab of its
environment, so run history never needs a separate ``lab`` column that could
drift out of sync with the catalogue.

Diagnostic probes deliberately belong to no lab: they are a cross-cutting
health check surfaced on the Diagnostics page.
"""

from typing import Any, Dict, List, Optional

DIAGNOSTICS = "diagnostics"

LABS: List[Dict[str, Any]] = [
    {
        "id": "foundations",
        "label": "Foundations",
        "tagline": "Tabular methods on small, fully observable worlds",
        "description": (
            "Gridworlds, chains, bandits and textbook MDPs where every state can be "
            "counted. Learning is fast enough to watch value estimates spread, which "
            "makes this the place to build intuition about exploration, bootstrapping "
            "and the discount factor."
        ),
        "accent": "violet",
        "icon": "grid",
        "match": {"categories": ["builtin", "toy-text", "custom"]},
        "algorithms": [
            "q_learning", "sarsa", "expected_sarsa", "double_q", "dyna_q",
            "mc_control", "ucb", "random", "dqn",
        ],
        "metrics": ["reward", "steps", "metric_td_error", "metric_epsilon", "metric_table_size"],
        "visual": "grid",
        "curriculum": [
            {"agent": "random", "env": "gridworld", "episodes": 50,
             "title": "Measure the floor",
             "why": "A random policy shows what zero skill looks like on this task."},
            {"agent": "q_learning", "env": "gridworld", "episodes": 300,
             "title": "Watch values propagate",
             "why": "Episode length should collapse from ~50 steps to the optimal 10."},
            {"agent": "sarsa", "env": "cliff_walking", "episodes": 400,
             "title": "On-policy caution",
             "why": "SARSA learns the safe path away from the cliff edge."},
            {"agent": "q_learning", "env": "cliff_walking", "episodes": 400,
             "title": "Off-policy optimism",
             "why": "Q-learning learns the risky edge path; compare it with SARSA."},
            {"agent": "dyna_q", "env": "dyna_maze", "episodes": 50,
             "title": "Plan with a model",
             "why": "Dyna-Q solves the maze in far fewer real episodes than Q-learning."},
            {"agent": "ucb", "env": "multi_armed_bandit", "episodes": 1000,
             "title": "Explore with confidence",
             "why": "Count-based bonuses beat epsilon-greedy without any tuning."},
        ],
    },
    {
        "id": "classic",
        "label": "Classic Control",
        "tagline": "Function approximation on low-dimensional dynamics",
        "description": (
            "CartPole, MountainCar, Acrobot, Pendulum and friends. States are a "
            "handful of continuous numbers, so neural networks replace tables. This is "
            "where deep RL algorithms are compared, debugged and tuned before they "
            "scale up."
        ),
        "accent": "cyan",
        "icon": "wave",
        "match": {"categories": ["gym", "continuous"], "exclude_tags": ["locomotion"]},
        "algorithms": [
            "dqn", "dqn_double", "dueling_dqn", "dqn_per", "c51", "reinforce",
            "reinforce_baseline", "a2c", "ppo", "ppo_continuous", "sac", "td3", "ddpg", "pets",
            "dreamer_lite", "random",
        ],
        "metrics": ["reward", "mean_reward", "steps", "metric_loss", "metric_entropy",
                    "metric_value_loss", "metric_epsilon"],
        "visual": "phase",
        "curriculum": [
            {"agent": "dqn", "env": "cartpole", "episodes": 400,
             "title": "Solve CartPole with DQN",
             "why": "The canonical first deep-RL result: mean return climbs to 500."},
            {"agent": "reinforce", "env": "cartpole", "episodes": 600,
             "title": "Pure policy gradient",
             "why": "No value function, high variance. Notice how noisy the curve is."},
            {"agent": "ppo", "env": "cartpole", "episodes": 300,
             "title": "Stabilise with PPO",
             "why": "Clipped updates and a critic make learning steady across seeds."},
            {"agent": "dqn", "env": "mountain_car", "episodes": 800,
             "title": "Sparse reward",
             "why": "Nothing happens for a long time, then momentum is discovered."},
            {"agent": "sac", "env": "pendulum", "episodes": 150,
             "title": "First continuous action",
             "why": "Torque instead of buttons: a stepping stone to MuJoCo."},
        ],
    },
    {
        "id": "continuous",
        "label": "Continuous Control",
        "tagline": "Torque control of simulated bodies",
        "description": (
            "MuJoCo locomotion and manipulation plus Box2D walkers. Actions are "
            "vectors of joint torques, observations are dozens of joint angles and "
            "velocities, and only off-policy actor-critic methods learn in a "
            "reasonable number of samples on a CPU."
        ),
        "accent": "amber",
        "icon": "joint",
        "match": {"categories": ["mujoco"], "tags": ["locomotion"]},
        "algorithms": ["sac", "td3", "ddpg", "ppo_continuous", "pets", "dreamer_lite", "random"],
        "metrics": ["reward", "mean_reward", "metric_critic_loss", "metric_actor_loss",
                    "metric_alpha", "metric_entropy", "metric_expl_noise"],
        "visual": "torque",
        "curriculum": [
            {"agent": "sac", "env": "inverted_pendulum", "episodes": 100,
             "title": "Check the pipeline",
             "why": "The easiest MuJoCo task. If SAC fails here, something is broken."},
            {"agent": "td3", "env": "reacher", "episodes": 200,
             "title": "Reach a target",
             "why": "Two joints, dense reward, a quick manipulation result."},
            {"agent": "sac", "env": "half_cheetah", "episodes": 300,
             "title": "Learn to run",
             "why": "No falling over, so the curve rises smoothly with experience."},
            {"agent": "sac", "env": "hopper", "episodes": 400,
             "title": "Balance and hop",
             "why": "Early termination on falling makes the curve much noisier."},
        ],
    },
    {
        "id": "perception",
        "label": "Perception",
        "tagline": "Learning to act from pixels",
        "description": (
            "MinAtar re-implements five Atari games on a 10x10 grid of binary "
            "feature channels: Atari's difficulty at a fraction of the compute. "
            "Observations are images, so representation learning is part of the "
            "problem."
        ),
        "accent": "rose",
        "icon": "pixels",
        "match": {"categories": ["minatar", "atari"]},
        "algorithms": ["dqn", "dqn_double", "dqn_per", "c51", "dueling_dqn", "ppo", "random"],
        "metrics": ["reward", "mean_reward", "steps", "metric_loss", "metric_epsilon",
                    "metric_buffer_size"],
        "visual": "frames",
        "curriculum": [
            {"agent": "random", "env": "minatar_freeway", "episodes": 20,
             "title": "Baseline the game",
             "why": "Random play on Freeway scores almost nothing."},
            {"agent": "dqn", "env": "minatar_freeway", "episodes": 300,
             "title": "DQN on Freeway",
             "why": "Freeway's dense progress signal makes early learning visible."},
            {"agent": "dqn_per", "env": "minatar_breakout", "episodes": 400,
             "title": "Prioritised replay",
             "why": "Rare brick-hits get replayed more often than empty frames."},
        ],
    },
    {
        "id": "multiagent",
        "label": "Multi-Agent",
        "tagline": "Two-player games against an opponent",
        "description": (
            "Turn-based board and card games from PettingZoo. The learner plays one "
            "seat against a scripted opponent, so reward is win, draw or loss and "
            "action masks rule out illegal moves."
        ),
        "accent": "lime",
        "icon": "seats",
        "match": {"categories": ["pettingzoo"]},
        "algorithms": ["ppo", "dqn", "dqn_double", "a2c", "random"],
        "metrics": ["reward", "mean_reward", "steps", "metric_loss", "metric_entropy"],
        "visual": "board",
        "curriculum": [
            {"agent": "random", "env": "tictactoe", "episodes": 200,
             "title": "Random vs random",
             "why": "The first player wins more often even with no skill."},
            {"agent": "dqn", "env": "tictactoe", "episodes": 1500,
             "title": "Never lose",
             "why": "With masks, DQN learns to block and fork against a random opponent."},
            {"agent": "ppo", "env": "connect_four", "episodes": 1500,
             "title": "Longer horizons",
             "why": "Connect Four needs planning several moves ahead."},
        ],
    },
    {
        "id": "language",
        "label": "Language models",
        "tagline": "RL fine-tuning of a tiny language model",
        "description": (
            "The techniques behind reasoning models such as DeepSeek-R1, at a scale that "
            "runs on a laptop: a small pretrained transformer, tasks with verifiable rewards, "
            "and the algorithms that fine-tune it: REINFORCE, GRPO and DPO, with supervised "
            "fine-tuning as the baseline. A hackable reward shows why the KL penalty exists."
        ),
        "accent": "sky",
        "icon": "text",
        "match": {"categories": ["language"]},
        "algorithms": ["grpo", "lm_reinforce", "dpo", "lm_sft"],
        "metrics": ["reward", "mean_reward", "metric_accuracy", "metric_eval_accuracy",
                    "metric_kl", "metric_entropy", "metric_zero_signal_groups", "metric_length"],
        "visual": "text",
        "curriculum": [
            {"agent": "lm_sft", "env": "lm_addition", "episodes": 150,
             "title": "The supervised baseline",
             "why": "Fine-tuning on correct answers: what RL has to beat, and needs labels."},
            {"agent": "grpo", "env": "lm_addition", "episodes": 150,
             "title": "Learn from a verifier",
             "why": "GRPO raises sampled accuracy with only a right/wrong signal."},
            {"agent": "lm_reinforce", "env": "lm_addition", "episodes": 150,
             "title": "One baseline for all",
             "why": "REINFORCE uses one batch-mean baseline; compare its noise with GRPO."},
            {"agent": "dpo", "env": "lm_sort", "episodes": 150,
             "title": "Learn from preferences",
             "why": "DPO improves from pairs of answers, with no reward in the loss."},
            {"agent": "grpo", "env": "lm_addition_hackable", "episodes": 150,
             "title": "Reward hacking",
             "why": "A flawed reward model pays for length; watch reward and accuracy part ways."},
        ],
    },
]

LAB_IDS = [lab["id"] for lab in LABS]


def get_lab(lab_id: str) -> Optional[Dict[str, Any]]:
    return next((lab for lab in LABS if lab["id"] == lab_id), None)


def lab_for_env(env: Dict[str, Any]) -> str:
    """Return the lab id an environment belongs to (or ``"diagnostics"``)."""
    category = env.get("category")
    if category == "probe":
        return DIAGNOSTICS
    tags = set(env.get("tags") or [])
    for lab in LABS:
        rule = lab["match"]
        if tags & set(rule.get("exclude_tags", [])):
            continue
        if category in rule.get("categories", []) or tags & set(rule.get("tags", [])):
            return lab["id"]
    return LABS[0]["id"]


def labs_with_members(envs: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> List[Dict]:
    """Lab definitions enriched with their environment names and counts."""
    out = []
    known_agents = {a["name"] for a in agents}
    for lab in LABS:
        members = [e for e in envs if e.get("lab") == lab["id"]]
        out.append({
            **{k: v for k, v in lab.items() if k != "match"},
            "envs": [e["name"] for e in members],
            "env_count": len(members),
            "available_env_count": sum(1 for e in members if e.get("available", True)),
            "algorithms": [a for a in lab["algorithms"] if a in known_agents],
        })
    return out
