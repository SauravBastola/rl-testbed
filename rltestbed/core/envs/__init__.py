"""Environment registry.

Builtin environments (gym / minatar / pettingzoo / pure-python) plus
auto-discovery of user-defined environments dropped into `custom_envs/`.

Custom env contract - a python file in custom_envs/ exposing:
    ENV_SPEC = {
        "name": "my_env",          # unique key
        "label": "My Env",
        "description": "...",
        "max_steps": 500,          # optional
        "obs_type": "vector",      # optional: vector | onehot | image
    }
    def make():
        return MyEnv()             # gym-like: reset()/step() -> 5-tuple

Observations may be flat lists/arrays or dicts {"obs": [...], "action_mask": [...]}.
Actions are discrete ints.

Probe environments are registered in a separate namespace and surfaced under
Benchmarks in the UI; they never appear inside the normal training catalog.
"""

import importlib.util
import sys
from typing import Any, Dict, List

from rltestbed.config import CUSTOM_ENVS_DIR
from .adapters import (
    GymAdapter,
    MinAtarAdapter,
    PettingZooSelfPlayAdapter,
    BuiltinAdapter,
)
from .base import EnvSpec, REGISTRY, gym_reward_threshold
from .builtin import GridworldEnv, CatchGameEnv, PROBES, TABULAR_ENVS


def _register(spec: EnvSpec, factory):
    REGISTRY[spec.name] = {"spec": spec, "factory": factory}


def _probe_dims(factory):
    """Instantiate an env once to record obs_dim/act_dim, or (None, None).

    Cheap for the pure-python builtins (no heavy dependencies are loaded), so
    we can do it at import time and let the catalog show real dimensions
    instead of blanks. Failures are non-fatal: the UI resolves lazily anyway.
    """
    try:
        env = factory()
        dims = (int(env.obs_dim), int(env.act_dim))
        env.close()
        return dims
    except Exception:
        return (None, None)


# --------------------------------------------------------------------- #
# Gymnasium classic control & box2d
# --------------------------------------------------------------------- #
_GYM = [
    dict(
        name="cartpole", env_id="CartPole-v1",
        description="Balance a pole on a moving cart. The canonical RL starter task: "
        "dense rewards, tiny state, fast episodes.",
        tags=["classic-control"], max_steps=500, difficulty="easy",
        obs_type="vector", reward_desc="+1 for every step the pole stays upright "
        "(max 500).", termination_desc="Pole falls >12°, cart leaves the track, or "
        "500 steps reached.", typical_episode_length=500,
        recommended_agents=["dqn", "dqn_double", "ppo", "a2c", "reinforce_baseline"],
        unsuitable_agents=["random"],
        beginner_exercise="Train DQN with default settings for 300-400 episodes. Watch "
        "the mean100 curve climb toward 500 - that is 'solved'.",
        advanced_exercise="Compare DQN vs PPO over 5 seeds each. Which is more stable? "
        "Measure wall-clock time to reach mean100 ≥ 475.",
        limitations=["Fully observable; no partial observability challenges."],
    ),
    dict(
        name="mountain_car", env_id="MountainCar-v0",
        description="Drive an underpowered car up a steep hill. The car cannot climb "
        "directly - it must build momentum by rocking, which requires long-horizon "
        "planning.",
        tags=["classic-control", "sparse-reward"], max_steps=200, difficulty="medium",
        obs_type="vector", sparse_reward=True,
        reward_desc="-1 per step until the goal is reached (reaching goal = 0).",
        termination_desc="Car reaches the flag on the right hill or 200 steps elapse.",
        typical_episode_length=200,
        recommended_agents=["dqn", "ppo"],
        unsuitable_agents=["q_learning", "sarsa"],
        beginner_exercise="Run DQN for 800+ episodes. Expect nothing for a long time - "
        "then a sudden jump when the agent discovers momentum. This teaches what sparse "
        "reward curves look like.",
        advanced_exercise="Compare epsilon_decay_steps=5k vs 30k. Does slower exploration "
        "find the solution more reliably?",
        limitations=["Reward signal is almost uninformative early; needs long exploration."],
    ),
    dict(
        name="lunar_lander", env_id="LunarLander-v2",
        description="Land a lunar module softly between two flags using side thrusters "
        "and a main engine (Box2D physics).",
        tags=["box2d"], max_steps=1000, difficulty="medium",
        obs_type="vector",
        reward_desc="Shaping rewards for landing softly between the flags; fuel use is "
        "penalised (+100..-100 range typical).",
        termination_desc="Crash, successful landing, or 1000 steps.",
        typical_episode_length=400,
        recommended_agents=["dqn", "ppo", "a2c"],
        beginner_exercise="PPO with default settings, ~600 episodes. Aim for mean eval "
        "reward above +200.",
        advanced_exercise="Sweep entropy_coef and compare crash rates across seeds.",
        limitations=["Requires the Box2D extra of gymnasium installed."],
    ),
    dict(
        name="acrobot", env_id="Acrobot-v1",
        description="Swing up a two-link pendulum by actuating the elbow. Sparse reward, "
        "requires energy pumping — a classic underactuated control benchmark.",
        tags=["classic-control", "sparse-reward"], max_steps=500, difficulty="medium",
        obs_type="vector", action_type="discrete",
        reward_desc="-1 per step until the tip reaches the line above the pivot.",
        termination_desc="Tip above threshold or 500 steps.",
        typical_episode_length=300,
        recommended_agents=["dqn", "ppo", "a2c"],
        unsuitable_agents=["q_learning", "sarsa"],
        beginner_exercise="DQN 600 episodes — watch for sudden drop in episode length when swing-up discovered.",
        advanced_exercise="Compare A2C vs PPO sample efficiency.",
        limitations=["Sparse; needs exploration."],
    ),
    dict(
        name="pendulum", env_id="Pendulum-v1",
        description="Balance an inverted pendulum by applying torque. Continuous torque in "
        "[-2,2] — prototypical continuous control.",
        tags=["classic-control", "continuous"], max_steps=200, difficulty="medium",
        obs_type="vector", action_type="continuous",
        reward_desc="Negative cost: -(theta² + 0.1·theta_dot² + 0.001·torque²).",
        termination_desc="200 steps (no early termination).",
        typical_episode_length=200,
        recommended_agents=["ddpg", "td3", "sac", "ppo"],
        unsuitable_agents=["dqn", "q_learning"],
        beginner_exercise="Train SAC for ~150 episodes. Mean return should climb from "
        "about -1200 toward -200 as the pendulum learns to swing up and hold.",
        advanced_exercise="Compare SAC, TD3 and DDPG over three seeds each. Which is "
        "least sensitive to the seed?",
        limitations=["Continuous action: discrete agents are blocked."],
    ),
    dict(
        name="bipedal_walker", env_id="BipedalWalker-v3",
        description="Make a bipedal robot walk by actuating 4 joints (Box2D). High-dimensional "
        "continuous control, tests sample efficiency and stability.",
        tags=["box2d", "continuous", "locomotion"], max_steps=1600, difficulty="hard",
        obs_type="vector", action_type="continuous",
        reward_desc="Reward for forward velocity minus torque + fall penalty.",
        termination_desc="Fall, 1600 steps, or out of bounds.",
        typical_episode_length=800,
        recommended_agents=["sac", "td3", "ppo"],
        unsuitable_agents=["dqn", "q_learning"],
        beginner_exercise="Train SAC for several hundred episodes and track how far the "
        "walker gets before falling.",
        advanced_exercise="Compare SAC vs TD3 on terrain robustness.",
        limitations=["Hard, Box2D, continuous — discrete agents unsuitable. Requires 1M+ steps."],
    ),
]

_GYM_URLS = {
    "cartpole": "https://gymnasium.farama.org/environments/classic_control/cart_pole/",
    "mountain_car": "https://gymnasium.farama.org/environments/classic_control/mountain_car/",
    "lunar_lander": "https://gymnasium.farama.org/environments/box2d/lunar_lander/",
    "acrobot": "https://gymnasium.farama.org/environments/classic_control/acrobot/",
    "pendulum": "https://gymnasium.farama.org/environments/classic_control/pendulum/",
    "bipedal_walker": "https://gymnasium.farama.org/environments/box2d/bipedal_walker/",
}

for _g in _GYM:
    _register(
        EnvSpec(
            name=_g["name"],
            label=_g["env_id"],
            category="gym",
            description=_g["description"],
            source="gymnasium",
            url=_GYM_URLS[_g["name"]],
            tags=_g["tags"],
            max_steps=_g["max_steps"],
            difficulty=_g["difficulty"],
            obs_type=_g["obs_type"],
            # action_type used to be dropped here, which registered Pendulum and
            # BipedalWalker as *discrete* - letting DQN "train" on them and
            # blocking SAC/TD3/DDPG.
            action_type=_g.get("action_type", "discrete"),
            sparse_reward=_g.get("sparse_reward", "sparse-reward" in _g["tags"]),
            requires=["box2d"] if "box2d" in _g["tags"] else [],
            reward_threshold=gym_reward_threshold(_g["env_id"]),
            # cheap for classic control; Box2D envs fail quietly when missing
            **dict(zip(("obs_dim", "act_dim"),
                       _probe_dims(lambda env_id=_g["env_id"]: GymAdapter(env_id)))),
            termination_desc=_g.get("termination_desc", ""),
            typical_episode_length=_g.get("typical_episode_length", _g["max_steps"]),
            recommended_agents=_g.get("recommended_agents", []),
            unsuitable_agents=_g.get("unsuitable_agents", []),
            limitations=_g.get("limitations", []),
            beginner_exercise=_g.get("beginner_exercise", ""),
            advanced_exercise=_g.get("advanced_exercise", ""),
        ),
        lambda env_id=_g["env_id"]: GymAdapter(env_id),
    )

# --------------------------------------------------------------------- #
# MinAtar arcade
# --------------------------------------------------------------------- #
_MINATAR_GAMES = {
    "minatar_breakout": ("breakout", "Breakout with compact pixel observations."),
    "minatar_space_invaders": ("space_invaders", "Space Invaders, compact."),
    "minatar_seaquest": ("seaquest", "Seaquest: dodge enemies, rescue divers, manage oxygen."),
    "minatar_freeway": ("freeway", "Freeway: race a chicken up lanes of traffic."),
    "minatar_asterix": ("asterix", "Asterix: collect treasure while avoiding hazards."),
}
for _name, (_game, _desc) in _MINATAR_GAMES.items():
    _register(
        EnvSpec(
            name=_name,
            label=f"MinAtar {(_game.replace('_', ' ')).title()}",
            category="minatar",
            description=_desc + " Compact 10×10 multi-channel binary pixels approximating "
            "the Atari Learning Environment - Atari-scale difficulty at Atari-fraction cost.",
            source="MinAtar",
            url="https://github.com/kenzay/MinAtar",
            tags=["arcade", "pixels"],
            requires=["minatar"],
            max_steps=3000,
            difficulty="hard",
            obs_type="image",
            reward_desc="Game-specific: points for breaking bricks / shooting enemies / "
            "progressing up lanes / collecting treasure.",
            termination_desc="Lives lost, oxygen depletion, timer expiry, or scenario end.",
            typical_episode_length=1500,
            recommended_agents=["dqn", "dqn_double", "dqn_per"],
            unsuitable_agents=["q_learning", "sarsa", "reinforce", "reinforce_baseline"],
            beginner_exercise="Start with MinAtar Freeway - its reward structure makes even "
            "small DQN runs visibly improve.",
            advanced_exercise="Benchmark DQN vs Double DQN on Breakout across 3 seeds; track "
            "loss stability as well as score.",
            limitations=[
                "Flattened 10x10xC pixel input; current MLP policies have no convolutional "
                "inductive bias - expect slow learning compared to CNN-based setups.",
            ],
        ),
        lambda g=_game: MinAtarAdapter(g),
    )

# --------------------------------------------------------------------- #
# PettingZoo self-play games (experimental)
# --------------------------------------------------------------------- #
_PZ = {
    "tictactoe": dict(
        label="Tic-Tac-Toe", steps=9, typical=5, requires=["pettingzoo"],
        description="Classic 3x3 noughts and crosses. Small enough to solve exactly, "
        "so a learner that loses to a random opponent has a bug.",
        recommended=["dqn", "ppo"],
        beginner="Train DQN for 1500 episodes. Mean return should rise above +0.8 "
        "(winning most games, rarely losing).",
        advanced="Compare learning as first player vs second player by changing the seat.",
    ),
    "connect_four": dict(
        label="Connect Four", steps=42, typical=20, requires=["pettingzoo"],
        description="Drop discs into a 7x6 grid; four in a row wins. Long enough "
        "horizons that a learner has to plan rather than react.",
        recommended=["ppo", "dqn"],
        beginner="PPO for 1500 episodes against the random opponent. Expect a win "
        "rate well above 90%.",
        advanced="Raise the difficulty by training against a stronger scripted "
        "opponent and see whether the curve still climbs.",
    ),
    "texas_holdem": dict(
        label="Texas Hold'em (Limit)", steps=400, typical=8,
        requires=["pettingzoo", "rlcard"],
        description="Two-player limit poker via RLCard. Hidden information and "
        "stochastic outcomes make the return signal very noisy.",
        recommended=["ppo"],
        beginner="Train PPO for 3000 episodes; average chip return should become "
        "positive against the random opponent.",
        advanced="Measure variance across 5 seeds; poker results need many games.",
    ),
}
for _name, _m in _PZ.items():
    _register(
        EnvSpec(
            name=_name,
            label=_m["label"],
            category="pettingzoo",
            description=_m["description"],
            source="pettingzoo",
            url=f"https://pettingzoo.farama.org/environments/classic/{_name}/",
            tags=["game", "two-player", "turn-based"],
            max_steps=_m["steps"],
            difficulty="medium" if _name != "texas_holdem" else "hard",
            action_masked=True,
            n_agents=2,
            requires=_m["requires"],
            reward_desc="The learner plays seat 0 against a uniformly random legal "
            "opponent. Reward is the game outcome from the learner's seat: +1 win, "
            "0 draw, -1 loss (chips for poker).",
            termination_desc="The game ends by its rules.",
            typical_episode_length=_m["typical"],
            recommended_agents=_m["recommended"],
            unsuitable_agents=["q_learning", "sarsa"],
            limitations=[
                "The opponent is scripted (random legal moves), not a second learner.",
                "Self-play against a frozen copy of the learner is planned.",
            ],
            beginner_exercise=_m["beginner"],
            advanced_exercise=_m["advanced"],
        ),
        lambda g=_name: PettingZooSelfPlayAdapter(g),
    )

# --------------------------------------------------------------------- #
# Builtin pure-python environments
# --------------------------------------------------------------------- #
_register(
    EnvSpec(
        name="gridworld",
        label="Gridworld",
        category="builtin",
        description="6×6 grid navigation: start top-left, reach the goal bottom-right. "
        "One-hot state observations make this the perfect first task for tabular methods.",
        source="testbed builtin",
        url="",
        tags=["navigation", "grid"],
        max_steps=200,
        difficulty="easy",
        obs_type="onehot",
        reward_desc="+1 on reaching the goal, −0.01 per step (encourages short paths).",
        termination_desc="Goal reached or 200 steps elapsed.",
        typical_episode_length=10,
        recommended_agents=["q_learning", "sarsa", "dqn"],
        unsuitable_agents=[],
        beginner_exercise="Tabular Q-learning, 300 episodes, watch td_error fall and the "
        "episode length collapse from ~50 steps to the optimal ~10.",
        advanced_exercise="Compare SARSA vs Q-learning final paths: SARSA learns a more "
        "cautious route near walls due to on-policy updates.",
        limitations=["Tiny state space - not representative of deep RL scaling."],
        **dict(zip(("obs_dim", "act_dim"),
                   _probe_dims(lambda: BuiltinAdapter(lambda: GridworldEnv(size=6))))),
    ),
    lambda: BuiltinAdapter(lambda: GridworldEnv(size=6)),
)

_register(
    EnvSpec(
        name="catch",
        label="Catch",
        category="builtin",
        description="Catch falling items by moving a paddle. A minimal arcade task that "
        "exercises delayed credit assignment without pixel complexity.",
        source="testbed builtin",
        url="",
        tags=["arcade", "simple"],
        max_steps=100,
        difficulty="easy",
        obs_type="vector",
        reward_desc="+1 catch, −1 miss, 0 otherwise.",
        termination_desc="Item lands (caught or missed) or 100 steps elapsed.",
        typical_episode_length=10,
        recommended_agents=["dqn", "dqn_double", "ppo", "a2c"],
        beginner_exercise="DQN for 200 episodes: expect mean eval reward climbing above +0.7.",
        advanced_exercise="Use Catch as your personal smoke test before any new experiment - "
        "if an algorithm can't learn Catch in minutes, debug before scaling.",
        limitations=["Three-dimensional observation is trivially learnable."],
        **dict(zip(("obs_dim", "act_dim"),
                   _probe_dims(lambda: BuiltinAdapter(CatchGameEnv)))),
    ),
    lambda: BuiltinAdapter(CatchGameEnv),
)

# --------------------------------------------------------------------- #
# Builtin tabular / exploration benchmarks (pure python)
# --------------------------------------------------------------------- #
_TABULAR = [
    dict(
        name="dyna_maze",
        label="Dyna Maze",
        description="Sutton & Barto's 6x9 maze with three walls and a single reward "
        "at the goal. Built to show how much planning with a learned model speeds "
        "up a tabular learner.",
        tags=["grid", "tabular", "planning", "sparse-reward"], max_steps=1000,
        difficulty="easy", obs_type="onehot",
        reward_desc="+1 on reaching the goal, 0 everywhere else.",
        termination_desc="Goal reached, or truncation at 1000 steps.",
        typical_episode_length=14,
        recommended_agents=["dyna_q", "q_learning"],
        unsuitable_agents=["random"],
        beginner_exercise="Run Dyna-Q with 0, 5 and 50 planning steps for 50 episodes "
        "each. With 50 the episode length drops to the optimal 14 within a handful "
        "of episodes; with 0 it takes dozens.",
        advanced_exercise="Compare steps-per-episode curves of Dyna-Q and Q-learning "
        "on the same seeds and count how many real environment steps each needs.",
        limitations=["Deterministic: Dyna-Q's last-outcome model is exact here, which "
                     "flatters it compared with stochastic worlds."],
    ),
    dict(
        name="nonstationary_bandit",
        label="Drifting Bandit",
        description="A 10-armed bandit whose true arm values take a small random walk "
        "after every pull. Whatever was best a while ago may not be best now, so "
        "exploration can never stop.",
        tags=["bandit", "exploration", "tabular", "nonstationary"], max_steps=1,
        difficulty="medium", obs_type="vector",
        reward_desc="Sampled from N(q_a, 1); every q_a drifts by N(0, 0.01) per pull.",
        termination_desc="One pull per episode.",
        typical_episode_length=1,
        recommended_agents=["ucb", "q_learning"],
        unsuitable_agents=["ppo", "a2c", "thompson"],
        beginner_exercise="Run UCB with step size 0 (sample averages) and 0.1 "
        "(constant). The constant step size keeps up with the drift; averages fall behind.",
        advanced_exercise="Compare Thompson sampling, which assumes stationarity, with "
        "constant-step UCB over 5,000 pulls.",
        limitations=["One state only: says nothing about generalisation."],
    ),
    dict(
        name="multi_armed_bandit",
        label="Multi-Armed Bandit",
        description="k arms, one state, Gaussian rewards. No generalisation is "
        "possible - every bit of performance must come from how well the agent "
        "balances trying new arms against exploiting the best one found so far.",
        tags=["bandit", "exploration", "tabular"], max_steps=1, difficulty="easy",
        obs_type="vector",
        reward_desc="Sampled from N(q_a, 1) where q_a is the true value of the "
        "chosen arm.",
        termination_desc="One pull per episode (the horizon is configurable).",
        typical_episode_length=1,
        recommended_agents=["q_learning", "sarsa", "dqn"],
        unsuitable_agents=["ppo", "a2c"],
        beginner_exercise="Run Q-learning with several epsilon values on the "
        "10-armed bandit and plot average reward. The classic curve - greedy is "
        "worst, a moderate epsilon is best, a large epsilon is bad - appears "
        "immediately.",
        advanced_exercise="Switch on the non-stationary variant: the true arm "
        "values drift, so a decaying epsilon eventually decays into permanent "
        "suboptimality. This is why schedules must be tuned per problem.",
        limitations=["No state at all - algorithms that rely on generalising "
                     "across states have nothing to work with here."],
    ),
    dict(
        name="river_swim",
        label="RiverSwim",
        description="Swim upstream against a current. Hugging the near bank pays a "
        "tiny guaranteed reward; the far bank pays 1.0 per step but is hard to "
        "reach. The standard benchmark for showing that naive epsilon-greedy "
        "exploration is not enough.",
        tags=["exploration", "tabular", "hard-exploration"], max_steps=100,
        difficulty="hard", obs_type="onehot",
        reward_desc="0.005 for swimming left from the start state; 1.0 for "
        "swimming right at the far bank; 0 otherwise.",
        termination_desc="No terminal state - episodes truncate at 100 steps.",
        typical_episode_length=100,
        recommended_agents=["q_learning", "dqn", "dqn_double"],
        unsuitable_agents=["random", "sarsa"],
        beginner_exercise="Train Q-learning with a generous epsilon (0.2+) for "
        "several thousand episodes and watch whether it ever reaches the far bank. "
        "At a low epsilon it usually never does.",
        advanced_exercise="Compare epsilon-greedy against optimistic initialisation "
        "(start the Q-values high). Optimism drives systematic exploration and "
        "typically solves RiverSwim far faster.",
        limitations=["Tiny state space, so the difficulty is purely exploratory - "
                     "it says nothing about function-approximation quality."],
    ),
    dict(
        name="n_chain",
        label="N-Chain",
        description="Walk a chain of states. Bailing out pays a small reward "
        "immediately; staying the course pays much more but only at the end of "
        "the chain. A clean test of long-horizon credit assignment.",
        tags=["credit-assignment", "tabular", "delayed-reward"], max_steps=100,
        difficulty="medium", obs_type="onehot",
        reward_desc="+2 for jumping back to the start; +10 for advancing while at "
        "the final state; 0 otherwise.",
        termination_desc="No terminal state - episodes truncate at 100 steps.",
        typical_episode_length=100,
        recommended_agents=["q_learning", "sarsa", "dqn"],
        unsuitable_agents=["random"],
        beginner_exercise="The greedy answer (+2 every step) beats the patient one "
        "for a long time. Train long enough and Q-learning should eventually "
        "overtake it.",
        advanced_exercise="Sweep the discount factor gamma. With gamma too low the "
        "optimal policy genuinely becomes 'bail out' - the cleanest demonstration "
        "that gamma encodes the agent's planning horizon.",
        limitations=["Deterministic apart from the slip probability."],
    ),
    dict(
        name="windy_gridworld",
        label="Windy Gridworld",
        description="Sutton & Barto's gridworld with an updraft that varies by "
        "column. Every move is displaced upward by the wind, so reaching the goal "
        "requires deliberately overshooting and letting the wind carry you in.",
        tags=["grid", "tabular", "control"], max_steps=200, difficulty="medium",
        obs_type="onehot",
        reward_desc="-1 per step until the goal is reached (undiscounted).",
        termination_desc="Reaching the goal cell, or truncation at 200 steps.",
        typical_episode_length=20,
        recommended_agents=["q_learning", "sarsa", "dqn"],
        unsuitable_agents=["random"],
        beginner_exercise="SARSA is the canonical algorithm here (Sutton & Barto "
        "Example 6.5). Expect the episode length to fall from ~200 toward ~15.",
        advanced_exercise="Enable the stochastic-wind variant (strength randomly "
        "-1/0/+1 per step) and measure how much longer convergence takes.",
        limitations=["70 states; tabular methods dominate and deep RL is overkill."],
    ),
]

for _t in _TABULAR:
    _factory = lambda cls=TABULAR_ENVS[_t["name"]]: BuiltinAdapter(cls)
    _obs_dim, _act_dim = _probe_dims(_factory)
    _register(
        EnvSpec(
            name=_t["name"],
            label=_t["label"],
            category="builtin",
            description=_t["description"],
            source="testbed builtin",
            url="",
            tags=_t["tags"],
            max_steps=_t["max_steps"],
            difficulty=_t["difficulty"],
            obs_type=_t["obs_type"],
            sparse_reward="sparse-reward" in _t["tags"],
            reward_desc=_t["reward_desc"],
            termination_desc=_t["termination_desc"],
            typical_episode_length=_t["typical_episode_length"],
            recommended_agents=_t["recommended_agents"],
            unsuitable_agents=_t["unsuitable_agents"],
            limitations=_t["limitations"],
            beginner_exercise=_t["beginner_exercise"],
            advanced_exercise=_t["advanced_exercise"],
            obs_dim=_obs_dim,
            act_dim=_act_dim,
        ),
        _factory,
    )

# --------------------------------------------------------------------- #
# Large Gymnasium suites: MuJoCo continuous control, toy-text tabular MDPs
# and the continuous classics.  Every entry is probed at import time, so a
# suite whose dependency is unavailable is skipped rather than crashing.
# --------------------------------------------------------------------- #
from .suites import build_suite_specs  # noqa: E402

SUITE_REGISTRATION: List[str] = build_suite_specs(EnvSpec, REGISTRY)

# --------------------------------------------------------------------- #
# Diagnostic probes - separate namespace, shown under Benchmarks only
# --------------------------------------------------------------------- #
_PROBE_META = {
    "probe1": dict(label="Probe 1 · Constant Reward"),
    "probe2": dict(label="Probe 2 · State Conditioning"),
    "probe3": dict(label="Probe 3 · Delayed Reward"),
    "probe4": dict(label="Probe 4 · Stochastic Reward"),
    "probe5": dict(label="Probe 5 · Two-Armed Bandit"),
    "probe6": dict(label="Probe 6 · Continuous Target", action_type="continuous", obs_type="vector",
                   recommended=["sac", "td3", "ddpg", "ppo_continuous"]),
}
for _pname, (_cls, _desc) in PROBES.items():
    _register(
        EnvSpec(
            name=_pname,
            label=_PROBE_META[_pname]["label"],
            category="probe",
            description=_desc + ". Diagnostic probe environment for verifying learning behaviour.",
            source="testbed builtin",
            url="",
            tags=["probe", "diagnostic"],
            max_steps=10,
            difficulty="easy",
            obs_type=_PROBE_META[_pname].get("obs_type", "onehot"),
            action_type=_PROBE_META[_pname].get("action_type", "discrete"),
            reward_desc="See benchmark description.",
            termination_desc="Single-step episode (or two steps for probe3).",
            typical_episode_length=2,
            recommended_agents=_PROBE_META[_pname].get(
                "recommended", ["dqn", "reinforce_baseline", "ppo"]),
            limitations=["Not a meaningful training task - diagnostics only."],
            **dict(zip(("obs_dim", "act_dim"), _probe_dims(lambda c=_cls: BuiltinAdapter(c)))),
        ),
        lambda c=_cls: BuiltinAdapter(c),
    )


# --------------------------------------------------------------------- #
# Custom env discovery
# --------------------------------------------------------------------- #
def discover_custom_envs(force: bool = False):
    """Import every module in custom_envs/ and register valid ENV_SPEC entries."""
    if not CUSTOM_ENVS_DIR.exists():
        return []
    loaded = []
    for path in sorted(CUSTOM_ENVS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"custom_envs.{path.stem}"
        try:
            if mod_name in sys.modules and not force:
                mod = sys.modules[mod_name]
            else:
                spec_ = importlib.util.spec_from_file_location(mod_name, path)
                mod = importlib.util.module_from_spec(spec_)
                sys.modules[mod_name] = mod
                spec_.loader.exec_module(mod)
            spec_dict = getattr(mod, "ENV_SPEC", None)
            make_fn = getattr(mod, "make", None)
            if not spec_dict or not callable(make_fn):
                continue
            name = spec_dict["name"]
            spec = EnvSpec(
                name=name,
                label=spec_dict.get("label", name),
                category="custom",
                description=spec_dict.get("description", ""),
                source=spec_dict.get("source", path.name),
                url=spec_dict.get("url", ""),
                tags=list(spec_dict.get("tags", [])) + ["custom"],
                max_steps=int(spec_dict.get("max_steps", 1000)),
                difficulty=spec_dict.get("difficulty", "experimental"),
                obs_type=spec_dict.get("obs_type", "vector"),
                action_type=spec_dict.get("action_type", "discrete"),
                lab=spec_dict.get("lab", ""),
                requires=list(spec_dict.get("requires", [])),
                reward_desc=spec_dict.get("reward_desc", ""),
                termination_desc=spec_dict.get("termination_desc", ""),
                limitations=spec_dict.get("limitations", []),
                beginner_exercise=spec_dict.get("beginner_exercise", ""),
                advanced_exercise=spec_dict.get("advanced_exercise", ""),
            )

            def _factory(fn=make_fn):
                env = fn()
                # accept either raw gym-like or already-unified envs
                if not hasattr(env, "obs_dim") or not hasattr(env, "act_dim"):
                    return BuiltinAdapter(lambda e=env: e)
                return env

            _register(spec, _factory)
            loaded.append(name)
        except Exception as e:
            print(f"[env discovery] failed to load {path.name}: {e}")
    return loaded


discover_custom_envs()


# Display order in the catalog: simple/tabular tasks first, then the deeper
# control and pixel suites, with user-supplied environments last.
_CATEGORY_ORDER = {
    "builtin": 0,
    "toy-text": 1,
    "gym": 2,
    "continuous": 3,
    "mujoco": 4,
    "minatar": 5,
    "pettingzoo": 6,
    "minigrid": 7,
    "custom": 8,
    "probe": 99,
}


def list_envs(resolve: bool = False, include_probes: bool = True) -> List[Dict[str, Any]]:
    """All environments sorted by category order.

    include_probes=False filters diagnostic probes out of the normal catalog.
    """
    specs = [entry["spec"].to_dict(resolve=resolve) for entry in REGISTRY.values()]
    if not include_probes:
        specs = [s for s in specs if s["category"] != "probe"]
    specs.sort(key=lambda s: (_CATEGORY_ORDER.get(s["category"], 9), s["name"]))
    return specs


def list_probe_envs() -> List[Dict[str, Any]]:
    return sorted(
        [e["spec"].to_dict() for e in REGISTRY.values() if e["spec"].category == "probe"],
        key=lambda s: s["name"],
    )


def get_env_spec(name: str) -> EnvSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown environment '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]["spec"]


def make_env(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown environment '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]["factory"]()
