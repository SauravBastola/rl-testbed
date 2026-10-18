# MuJoCo + continuous control

This document explains the MuJoCo environment suite, why the testbed needs
its own runtime compatibility shim, and how to get the continuous-control
algorithms (DDPG, TD3, SAC) producing real learning curves.

## What's installed

The testbed pins these versions in `requirements.txt`:

```
torch==2.4.0
gymnasium==0.29.1
mujoco==3.10.0
```

`mujoco==3.10.0` provides the C-based MuJoCo engine. Gymnasium 0.29.1
bundles the 11-task MuJoCo locomotion/manipulation suite (Ant, HalfCheetah,
Hopper, Walker2d, Swimmer, Humanoid, HumanoidStandup, InvertedPendulum,
InvertedDoublePendulum, Reacher, Pusher) at the `v4` Gymnasium revision.

The engine itself needs no MuJoCo licence since DeepMind open-sourced it in
2022.

## Why a compat shim is needed

Two of the dependencies that this project pins (for the sake of
reproducibility and because the existing adapters and benchmarks are
verified against them) have moved forward and broken specific
Gymnasium-0.29.1 internals. Rather than touch site-packages, the testbed
applies two small runtime patches in `rltestbed/compat.py`:

1. **`np.float_` was removed in NumPy 2.0**, but
   `gymnasium/envs/classic_control/acrobot.py` still uses it inside its
   RK4 integrator. Without the patch, every `Acrobot-v1.step()` raises
   `AttributeError`.
2. **MuJoCo ≥ 3.0 rejects the legacy `collision="predefined"` attribute**
   that Gymnasium 0.29.1 still ships in `swimmer.xml`. Without the patch,
   `gym.make("Swimmer-v4")` refuses to build the model.

Both patches are idempotent, reversible, and run automatically on
`import rltestbed`. To see the diagnostic output:

```python
from rltestbed.compat import apply_compat_shims, mujoco_version
print(apply_compat_shims())     # {'applied': True, 'mujoco_assets_patched': 1}
print(mujoco_version())          # '3.10.0'
```

## How the adapters work with continuous actions

`rltestbed/core/envs/adapters.py::GymAdapter` detects a `Box` action space
and exposes two extras to the trainer:

- `act_dim` = total number of action scalars (e.g. 8 for Ant).
- `act_low`, `act_high` = the per-dimension Box bounds, as flat float arrays.

The trainer calls `agent.set_action_bounds(act_low, act_high)` on every
constructed agent. Continuous agents store these bounds and use them in two
places:

- `get_action(obs)` returns the **environment** action: the agent's
  normalised `[-1, 1]` output is rescaled to `[act_low, act_high]` before
  being handed to `env.step()`.
- `update(obs, action, reward, next_obs, done)` receives the
  environment-scaled action back from the trainer and un-scales it to
  `[-1, 1]` for storage in the replay buffer. Critics and target networks
  always see bounded inputs.

The action marshalling is fully encapsulated in
`_ContinuousBase._to_env_action` and `_ContinuousBase._to_norm_action` in
`rltestbed/core/agents/continuous.py`.

## Recommended starting point

```bash
# SAC is the most robust default for a new continuous task.
rltestbed train sac half_cheetah --episodes 200
```

On HalfCheetah-v4 with the default hyperparameters you should see clear
learning in the first 60 episodes and an evaluation return north of 3 000
by episode 150 (150 000 environment steps).

## Tuning notes

The continuous agents all share a common parameter set plus a few
algorithm-specific ones. The most useful knobs, in roughly the order you
should reach for them:

| Knob | Where it lives | Why it matters |
|---|---|---|
| `normalize_obs` | common | MuJoCo observations mix very different scales. Leaving this **on** is the single biggest stability improvement. |
| `buffer_size` | common | Defaults to 100 000 to keep memory sane even for Humanoid's 376-dim observations. Raise it if you have RAM to spare. |
| `gamma` | common | 0.99 is fine for locomotion; 0.995 sometimes helps for tasks with very delayed reward. |
| `tau` | common | 0.005 is the paper default. Lower = more stable targets; higher = faster tracking. |
| `lr` / `critic_lr` | common | 3e-4 is a safe default. If loss explodes, drop both by 10×. |
| `warmup_steps` | common | Random data before learning starts. 1 000 is a good default; raise it for Humanoid. |
| `auto_entropy` (SAC) | SAC only | Leave **on**. The single biggest "SAC is magic" lever. |
| `policy_delay` (TD3) | TD3 only | 2 is the paper default. Raising it to 3–4 is a common stability knob. |
| `target_noise` / `noise_clip` (TD3) | TD3 only | Smoothing noise on the target action. Raising `target_noise` makes the target more conservative. |

## Headless rendering for environments that need a display

If you are running the trainer on a headless box (server, CI, container)
and you need to call `env.render(mode="rgb_array")`, set:

```bash
export MUJOCO_GL=osmesa    # CPU-rendered OpenGL
# or
export MUJOCO_GL=egl       # GPU-rendered OpenGL if a GPU is available
```

The defaults `glfw` and `egl` require an X server or a GPU respectively.

## Where the code lives

- `rltestbed/compat.py` — the two runtime patches.
- `rltestbed/core/envs/suites.py` — the MuJoCo / toy-text / continuous
  catalogue (one dictionary per environment, with reward description and
  recommended agents).
- `rltestbed/core/envs/adapters.py` — the `GymAdapter` that exposes
  `act_dim`, `act_low`, `act_high` to the trainer.
- `rltestbed/core/agents/continuous.py` — DDPG, TD3 and SAC plus their
  shared base, replay buffer and observation normaliser.
- `rltestbed/core/agents/base.py` — the `set_action_bounds` hook that
  makes action-bounds propagation uniform across all agents.
