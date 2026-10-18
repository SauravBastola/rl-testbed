# Environments

Every environment registered with the testbed is described here by **family**.
Each family shares installation, spaces, recommended algorithms, and beginner
/ advanced exercises; per-environment specifics live in the web app's
**Environment detail** page (the metadata comes from
`rltestbed/core/envs/` and is the single source of truth).

## Table of contents

- [Gymnasium classic control](#gymnasium-classic-control) — 6 envs
- [Gymnasium toy text](#gymnasium-toy-text) — 4 envs
- [Gymnasium continuous classics](#gymnasium-continuous-classics) — 1 env
- [MuJoCo continuous control](#mujoco-continuous-control) — 11 envs
- [MinAtar arcade](#minatar-arcade) — 5 envs
- [PettingZoo self-play (experimental)](#pettingzoo-self-play) — 2 envs
- [Built-in pure-python](#built-in-pure-python) — 6 envs
- [Diagnostic probes](#diagnostic-probes) — 5 envs
- [Adding your own](#adding-your-own)

Total: **36 trainable environments + 5 diagnostic probes**, in addition to
anything dropped into `custom_envs/`.

---

## Gymnasium classic control

Six dense-reward, small-observation tasks — the right place to start. All
are `gym.vector` observations and discrete actions, with continuous `Box`
action variants for `Pendulum` and `MountainCarContinuous`.

| Env | Obs · Act | Difficulty | Best with |
|---|---|---|---|
| `cartpole`           | 4 · 2  discrete | easy | DQN, PPO, A2C, REINFORCE+baseline |
| `mountain_car`       | 2 · 3  discrete | medium (sparse) | DQN, PPO |
| `acrobot`            | 6 · 3  discrete | medium (sparse) | DQN, PPO, A2C |
| `lunar_lander`       | 8 · 4  discrete | medium | DQN, PPO, A2C |
| `pendulum`           | 3 · 1  continuous | medium | SAC, TD3, DDPG, PPO |
| `bipedal_walker`     | 24 · 4 continuous | hard | SAC, TD3, PPO |

**Beginner exercise** — train DQN on CartPole to mean-100 ≥ 475. This is the
canonical sanity check that the testbed, the agent, and your environment are
all wired correctly.

**Advanced exercise** — sweep PPO's entropy coefficient on LunarLander and
plot crash rate across 5 seeds.

---

## Gymnasium toy text

Small tabular MDPs with discrete observation spaces. Perfect for tabular
methods and for teaching concepts that get buried in deep-RL noise.

| Env | Obs · Act | Difficulty | Why it's interesting |
|---|---|---|---|
| `frozen_lake`    | 16 · 4 | easy | Stochastic slippery ground — the cleanest demo of why value iteration needs optimistic exploration. |
| `cliff_walking`  | 48 · 4 | easy | The shortest path is dangerous. SARSA learns the safe route; Q-learning learns the risky one. |
| `taxi`           | 500 · 6 | medium | Multi-fact state (position + passenger + destination). |
| `blackjack`      | Tuple (player, dealer, usable ace) | medium | Tuple observation space — the adapter one-hot-encodes each component. |

---

## Gymnasium continuous classics

| Env | Obs · Act | Difficulty | Notes |
|---|---|---|---|
| `mountain_car_continuous` | 2 · 1 in [-1, 1] | medium | Continuous-torque Mountain Car. Action is penalised — a thrashing policy loses to a gentle one even if both reach the flag. |

---

## MuJoCo continuous control

Eleven tasks, all with continuous `Box` action spaces and vector
observations. **Requires the continuous-control algorithms (DDPG / TD3 /
SAC)**; discrete agents are blocked by `rltestbed/capabilities.py`.

Tested with the MuJoCo engine shipped at `mujoco==3.10.0` and Gymnasium
`0.29.1`. A small runtime shim in `rltestbed/compat.py` patches the Swimmer
XML and the `np.float_` reference in Acrobot so the pinned environment
versions keep working as the dependencies move on.

### CartPoles

| Env | Obs · Act | Difficulty | Why bother |
|---|---|---|---|
| `inverted_pendulum`       | 4 · 1 in [-3, 3]  | easy | Continuous-force CartPole. Easiest sanity check for the continuous-control pipeline. |
| `inverted_double_pendulum` | 11 · 1 in [-1, 1] | medium | Two-link pendulum on a cart — the second link makes the dynamics genuinely harder. |

### Arms

| Env | Obs · Act | Difficulty |
|---|---|---|
| `reacher` | 11 · 2 in [-1, 1] | easy |
| `pusher`  | 23 · 7 in [-2, 2] | hard |

Reacher is the friendliest manipulation task: short episodes, dense reward.
Pusher is harder because the reward is a sum of three competing distances.

### 2D runners

| Env | Obs · Act | Difficulty | Notes |
|---|---|---|---|
| `half_cheetah` | 17 · 6 in [-1, 1] | medium | The standard sample-efficiency benchmark. |
| `hopper`       | 11 · 3 in [-1, 1] | hard | Adds an "unhealthy" termination state; bad policies fall over rather than just scoring badly. |
| `walker2d`     | 17 · 6 in [-1, 1] | hard | 2D biped with twice Hopper's joints. |

### Swimmers / Quadruped / Humanoids

| Env | Obs · Act | Difficulty |
|---|---|---|
| `swimmer`  | 8 · 2 in [-1, 1] | hard (exploration trap) |
| `ant`      | 27 · 8 in [-1, 1] | hard |
| `humanoid` | 376 · 17 in [-0.4, 0.4] | expert |
| `humanoid_standup` | 376 · 17 in [-0.4, 0.4] | expert |

`swimmer` is a classic *exploration trap* — its weak gradients make
single-seed results meaningless and it is the standard stress test for
exploration in continuous control. `humanoid` and `humanoid_standup` are
the most expensive tasks in the suite (376-dim observations, 17-dim
actions); only attempt them after Ant works.

### Recommended pairings

| Env | Try first | Then try |
|---|---|---|
| `inverted_pendulum`     | SAC | TD3 |
| `inverted_double_pendulum` | SAC | TD3 |
| `reacher`               | SAC | TD3 |
| `pusher`                | SAC | TD3 |
| `half_cheetah`          | SAC | TD3 |
| `hopper`                | SAC | TD3 |
| `walker2d`              | SAC | TD3 |
| `swimmer`               | SAC (entropy matters here) | TD3 |
| `ant`                   | SAC | TD3 |
| `humanoid`              | SAC | (on-policy methods with parallel envs) |
| `humanoid_standup`      | SAC | — |

---

## MinAtar arcade

Compact 10×10 multi-channel binary-pixel approximations of the Atari
Learning Environment — Atari-scale difficulty at a fraction of the compute.
`flattens` the pixels for the current MLP policies.

| Game | Notes |
|---|---|
| `minatar_breakout`        | Standard brick-breaker. |
| `minatar_space_invaders`  | Shoot descending aliens. |
| `minatar_seaquest`        | Dodge enemies, manage oxygen, rescue divers. |
| `minatar_freeway`         | Race a chicken up lanes of traffic. |
| `minatar_asterix`         | Collect treasure while avoiding hazards. |

All have `obs_type=image` (flattened before reaching the agent). Use DQN or
Double DQN; expect learning to be slow without a convolutional backbone.

---

## PettingZoo self-play (experimental)

Two-player games where **one network controls every seat**. Marked
experimental because controlling both sides is a simplification of true
self-play.

| Game | Notes |
|---|---|
| `tictactoe`     | Action-mask friendly; small enough to inspect failure modes by hand. |
| `texas_holdem`  | Limit Texas Hold'em via RLCard/PettingZoo. |

The rewards accumulate across seats — a clean win-rate metric needs a
custom wrapper. Use DQN to start.

---

## Built-in pure-python

Six dependency-free environments: tiny, fast, and designed to isolate a
specific learning phenomenon. Excellent for first-episode smoke tests and
for teaching the classic tabular ideas.

| Env | Obs · Act | Why it exists |
|---|---|---|
| `gridworld`             | 64 · 4  | The starter task. One-hot states make it ideal for Q-learning. |
| `catch`                 | 3 · 3   | Minimal arcade game that exercises delayed credit assignment without pixel complexity. |
| `multi_armed_bandit`    | 1 · k   | Pure exploration-exploitation with no state at all. |
| `river_swim`            | 6 · 2   | The hard-exploration benchmark. Naive ε-greedy gets stuck. |
| `n_chain`               | 5 · 2   | Long-horizon credit assignment: the myopic answer pays first, the patient one wins. |
| `windy_gridworld`       | 70 · 4  | Sutton & Barto Example 6.5: an updraft varies by column. |

The four tabular environments (`multi_armed_bandit`, `river_swim`,
`n_chain`, `windy_gridworld`) are pure python and live in
`rltestbed/core/envs/builtin/tabular.py`.

---

## Diagnostic probes

Five small, single-decision tasks with *known correct behaviour*. Use them
to verify that a new algorithm implementation can actually learn the
simplest things before you trust it on harder tasks.

| Probe | What it tests |
|---|---|
| `probe1` | Constant reward — any update should converge. |
| `probe2` | State-conditioned reward — a value-based agent must learn V(s). |
| `probe3` | Two-step delayed reward — credit assignment over a short horizon. |
| `probe4` | Stochastic reward — expectation maximisation under noise. |
| `probe5` | Two-armed bandit — basic policy preference. |

Probes are not surfaced in the normal training catalog; find them under the
**Benchmarks** page.

---

## Adding your own

Drop a Python file into `custom_envs/` exposing `ENV_SPEC` + `make()`:

```python
ENV_SPEC = {
    "name": "my_env",
    "label": "My Environment",
    "description": "…",
    "max_steps": 500,
    "obs_type": "vector",          # or "onehot" or "image"
    "tags": ["experimental"],
}
def make():
    return MyEnv()                 # gym-like: reset()/step() -> 5-tuple
```

See `custom_envs/example_env.py` for a working example. New files are
picked up on the next server restart.
