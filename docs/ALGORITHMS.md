# Algorithms

Reference for every algorithm currently implemented in RL Testbed. Each entry
follows the same structure: intuition, when to use it, the update rule in math
and pseudocode, the hyperparameters exposed in the UI, and the failure modes
the algorithm is most prone to. Implemented agents live in
`rltestbed/core/agents/` and are registered in
`rltestbed/core/agents/__init__.py`.

## Table of contents

- [Value-based](#value-based)
  - [Q-learning](#q-learning)
  - [SARSA](#sarsa)
  - [DQN](#dqn)
  - [Double DQN](#double-dqn)
  - [Dueling DQN](#dueling-dqn)
- [Policy-gradient](#policy-gradient)
  - [REINFORCE](#reinforce)
  - [REINFORCE with baseline](#reinforce-with-baseline)
- [Actor-critic (discrete)](#actor-critic-discrete)
  - [A2C](#a2c)
  - [PPO](#ppo)
- [Continuous control](#continuous-control)
  - [DDPG](#ddpg)
  - [TD3](#td3)
  - [SAC](#sac)
- [Baseline](#baseline)
  - [Random](#random)
- [Roadmap](#roadmap)

---

## Value-based

### Q-learning
**Status:** implemented · **On/off-policy:** off · **Obs:** onehot · **Act:** discrete

The tabular workhorse. Learns the value of taking action `a` in state `s` and
then greedily following the best value, regardless of which policy actually
generated the data. Off-policy convergence is the key feature — you can learn
an optimal policy from exploration noise, or even from a hand-coded policy.

`Q(s, a) ← Q(s, a) + α · ( r + γ · max_a' Q(s', a') − Q(s, a) )`

Best on the smallest tabular tasks: `gridworld`, `frozen_lake`,
`cliff_walking`, `taxi`, `n_chain`. Will not scale to vectors or images.

### SARSA
**Status:** implemented · **On/off-policy:** on · **Obs:** onehot · **Act:** discrete

Same update as Q-learning but bootstraps from the action the *current* policy
actually took, not the greedy one. The difference is invisible on deterministic
environments and dramatic on stochastic ones — `cliff_walking` is the textbook
example: SARSA learns the safer long path, Q-learning learns the shorter risky
one.

`Q(s, a) ← Q(s, a) + α · ( r + γ · Q(s', a') − Q(s, a) )`, with `a' ~ π(·|s')`.

### DQN
**Status:** implemented · **On/off-policy:** off · **Obs:** vector / onehot · **Act:** discrete · **Action-mask:** ✔

Approximates `Q(s, a)` with a neural network, stores transitions in a replay
buffer, and uses a target network (hard-synced every N steps) to stabilise
bootstrapping. Epsilon-greedy exploration with linear decay.

- Replay buffer: uniform 50 000 (configurable).
- Target update: hard copy every `target_update_freq` gradient steps (default 200).
- Loss: Huber of the one-step TD target.
- Gradient clipping at norm 10.

Works on `cartpole`, `catch`, `lunar_lander`, `acrobot`, `mountain_car`.

### Double DQN
**Status:** implemented · **On/off-policy:** off · **Obs:** vector / onehot · **Act:** discrete · **Action-mask:** ✔

DQN with the decoupled action-selection / action-evaluation step: the online
network *picks* the next action, the target network *scores* it. Same replay
buffer and target sync as DQN.

`a* = argmax_a Q(s', a)` (online);  `y = r + γ · Q_target(s', a*)`.

Reduces the systematic overestimation of vanilla DQN. Almost always preferred
over DQN when both are available.

### Dueling DQN
**Status:** implemented · **On/off-policy:** off · **Obs:** vector / onehot · **Act:** discrete · **Action-mask:** ✔

Same training loop as DQN but the Q-network is split into a value stream
`V(s)` and an advantage stream `A(s, a)`, combined as
`Q = V + (A − mean_a A)`. Helps in states where the action choice does not
matter — the value stream can be learned without wasting capacity on
distinguishing actions that lead to the same outcome.

---

## Policy-gradient

### REINFORCE
**Status:** implemented · **On/off-policy:** on · **Obs:** vector / onehot · **Act:** discrete

The classic policy-gradient theorem applied directly: sample a full episode,
multiply the log-probability of each action by the return from that point on,
and update. High variance, no baseline, usually needs a baseline to learn
anything non-trivial. `reinforce_baseline` should be your default.

`∇θ J(θ) = 𝔼[ ∇θ log πθ(a|s) · G_t ]`

### REINFORCE with baseline
**Status:** implemented · **On/off-policy:** on · **Obs:** vector / onehot · **Act:** discrete

REINFORCE with a learned value baseline `V(s)` subtracted from the return.
The baseline dramatically reduces variance without changing the expected
gradient (the baseline is independent of the action).

`∇θ J(θ) = 𝔼[ ∇θ log πθ(a|s) · (G_t − V(s)) ]`

Works on `cartpole`, `lunar_lander`, `catch`, `acrobot` with default
hyperparameters.

---

## Actor-critic (discrete)

### A2C
**Status:** implemented · **On/off-policy:** on · **Obs:** vector / onehot · **Act:** discrete

Advantage Actor-Critic. A single shared network produces a policy head and a
value head; the n-step advantage `A = R − V(s)` bootstraps the value
estimator and is used to weight the policy update. Synchronous, simpler than
PPO, but more brittle.

`L_actor = −log π(a|s) · A`;  `L_critic = (R − V(s))²`.

### PPO
**Status:** implemented · **On/off-policy:** on · **Obs:** vector / onehot · **Act:** discrete

Proximal Policy Optimization. The same actor-critic split as A2C, but the
policy update is clipped to stay close to the previous policy, which keeps
the optimisation well-behaved across many epochs of minibatch updates.

`L_clip = 𝔼[ min( r(θ)·A, clip(r(θ), 1−ε, 1+ε)·A ) ]`

The most reliable on-policy algorithm in the catalogue. Recommended for
`cartpole`, `lunar_lander`, `catch` when you want a strong default.

---

## Continuous control

These three algorithms are required for the MuJoCo suite, where every
environment has a continuous `Box` action space. All three emit actions in
a normalised `[-1, 1]` range and rescale them to the environment's real
limits via `Agent.set_action_bounds` (the trainer calls this hook with
`env.act_low`/`env.act_high`).

All three share:
- a **numpy-backed replay buffer** with actions stored in `[-1, 1]`,
- **Polyak soft target updates** with coefficient `tau`,
- **observation normalisation** (Welford mean/var) — on by default because
  MuJoCo observations mix positions, angles, velocities and contact forces
  with wildly different scales.

### DDPG
**Status:** implemented · **On/off-policy:** off · **Obs:** vector · **Act:** continuous

Deep Deterministic Policy Gradient. A deterministic actor is trained by
ascending the critic's gradient: `max_θ Q(s, μ_θ(s))`. Exploration comes
from additive Gaussian noise on the executed action.

- Single Q-network, no twin critic (this is what TD3 fixes).
- Target networks for both actor and critic, soft-updated every gradient step.
- Recommended for: `inverted_pendulum`, `reacher`, `pendulum`,
  `mountain_car_continuous`. Avoid for `humanoid`, `swimmer`.

### TD3
**Status:** implemented · **On/off-policy:** off · **Obs:** vector · **Act:** continuous

Twin Delayed DDPG. Three targeted fixes on top of DDPG that together remove
its overestimation bias and most of its hyperparameter fragility:

1. **TWIN CRITICS** — train two Q-networks, take the `min` for the target.
2. **DELAYED POLICY UPDATES** — only update the actor and target networks
   every `policy_delay` gradient steps (default 2), so the policy never
   chases a value estimate that is still moving.
3. **TARGET POLICY SMOOTHING** — add small clipped Gaussian noise to the
   target action when computing the TD target, so a narrow spike in the
   critic cannot be exploited.

`y = r + γ · (1 − done) · min_i Q̄_i(s', clip(μ̄(s') + clip(N(0, σ), −c, c), −1, 1))`

### SAC
**Status:** implemented · **On/off-policy:** off · **Obs:** vector · **Act:** continuous

Soft Actor-Critic. The policy is trained to maximise *entropy-regularised*
return: reward plus a bonus proportional to how random the policy still is.
With `auto_entropy` on (default), the temperature `α` is tuned automatically
so the policy's entropy matches a target (`-act_dim` by default). That
removes the single most annoying SAC hyperparameter.

`y = r + γ(1−done)·( min_i Q̄_i(s', a') − α·log π(a'|s') )`,  `a' ~ π(·|s')`

Why it usually wins on a new continuous task: the entropy bonus keeps the
policy exploring, the twin critics curb overestimation, and the soft targets
prevent the value estimate from racing ahead. Recommended as the **default
choice** for every MuJoCo environment in the suite.

---

## Baseline

### Random
**Status:** implemented · **Obs:** any · **Act:** discrete or continuous

A uniform random policy. Useful as a sanity check (your agent should beat
random) and as a warmup data source for off-policy algorithms.

---

## Roadmap

The following are intentionally listed as **planned** in the registry — the
UI shows them but refuses to train them, because none of them are
implemented yet.

| Algorithm | What it would add | Difficulty to implement |
|---|---|---|
| `expected_sarsa` | Softens max in tabular control; closes the on/off gap | low |
| `dqn_per` | Prioritised experience replay | medium |
| `noisy_dqn` | Parametric noise replaces ε-greedy | medium |
| `c51` | Distributional Bellman (categorical 51) | medium |
| `trpo` | Trust-region natural gradient; PPO is the practical substitute | high |
| `dyna_q` | Model-based acceleration via learned world model | high |
| `independent_learners` | Per-seat Q/policy for multi-agent PettingZoo | medium |

To contribute a new algorithm: subclass `rltestbed.core.agents.base.Agent`,
define a `SPEC` with `build_spec(...)`, and register the class in
`AGENT_REGISTRY` in `rltestbed/core/agents/__init__.py`.
