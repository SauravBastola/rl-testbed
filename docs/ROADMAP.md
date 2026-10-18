# Roadmap

Each item lists what "done" means. Items are grouped by the release that should
carry them; within a release they are roughly in priority order. Update this file
in the same commit that finishes an item.

## v3.1: experimentation at scale (this release)

- [x] **Hyperparameter sweeps.** Grid or random search over any algorithm's
  parameters, several seeds per configuration, queued through the job manager.
  Done when a sweep can be launched from the UI, its configurations are ranked by
  mean final return ± std, and the best configuration opens pre-filled in New run.
- [x] **Per-parameter effect view.** For every swept parameter, mean result per
  value, so "which knob mattered" is visible without reading a table.
- [x] **Continuous-action PPO.** Gaussian policy; runs on Pendulum and MuJoCo.
- [x] **Continuous probe.** A one-step continuous-action diagnostic, so SAC, TD3,
  DDPG and Gaussian PPO appear in the Diagnostics matrix.
- [x] **CI and task runner.** GitHub Actions runs the same checks as `make check`.

## v3.2: richer experiments

- [ ] **Experiment files.** `rltestbed run experiment.yaml` describing runs or a
  sweep; the UI can export any run or sweep as such a file. Done when an exported
  file reproduces the same runs on another machine.
- [ ] **Run export.** CSV/JSON download of a run's history and of a comparison.
- [ ] **Sweep early stopping.** Successive halving: stop the worst half of
  configurations at fixed checkpoints. Done when a sweep uses measurably less
  compute than the full grid and still finds the same best configuration on CartPole.
- [ ] **Resume from checkpoint.** Restart continues from `checkpoint.pth` instead of
  episode 0 (agent state, optimiser, replay buffer optional).

## v3.3: breadth of methods

- [ ] **Self-play.** The game adapter accepts a frozen copy of the learner as the
  opponent, refreshed every N episodes. Win rate against the random opponent is
  reported alongside.
- [ ] **CNN encoder.** Optional convolutional torso for pixel observations
  (MinAtar first). The flattened-pixels warning disappears for agents using it.
- [ ] **NoisyNet, n-step returns, then Rainbow** composed from the existing parts.
- [ ] **Offline RL pipeline.** Record a dataset from a kept model; train
  Behaviour Cloning and CQL from it. Needs a `datasets` table and a dataset page.
- [ ] **Intrinsic motivation.** A reward-wrapper hook in the trainer; RND on top.

## v3.4: breadth of environments

- [ ] **MiniGrid** as an optional extra, in a new "Exploration" lab (sparse
  reward, partial observability). `availability.py` already knows the extra.
- [ ] **Atari (ALE)** as an optional extra in the Perception lab, with frame
  stacking. Depends on the CNN encoder.
- [ ] **Procedural environments** for generalisation (train/test seed splits).

## v4: operations

- [ ] **Docker image** with CPU PyTorch and every optional extra.
- [ ] **Remote workers.** Job manager dispatches to other machines over SSH;
  runs keep their local artifact layout.
- [ ] **GPU scheduling.** Per-run device selection respects available GPUs and
  the concurrency limit counts per device.
- [ ] **Postgres option** behind the same registry interface, for many users.

## Ongoing

- Every new algorithm passes at least one probe before merge.
- Every new page has a smoke test and loading, empty and error states.
- `make check` passes on every commit to `main`.
