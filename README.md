# RL Testbed

**A local laboratory for learning and doing reinforcement learning.**
Pick an environment and an algorithm, launch a run, watch it learn live, compare
methods across seeds, and read the theory behind every one of them, in one web
app with a matching command line.

![Classic Control lab](docs/screenshots/lab_classic--midnight.png)

Every algorithm is implemented in this repository, readably, from tabular
Q-learning to SAC, PETS world models and GRPO for language models, so you can
study the code as well as run it.

## What you can do

- **Train and watch.** 29 algorithms on 43 environments, each run in its own
  process with a live learning curve, greedy evaluations and per-algorithm metrics.
- **Compare honestly.** Run several seeds and see the mean with a ±1σ band;
  sweep hyperparameters and see each one's effect.
- **See what the agent sees.** Step through episodes frame by frame, with the
  observation labelled and the chosen action named. World models show their
  imagined trajectory against the real one.
- **Learn.** A 23-step learning path from a random baseline to RL for language
  models; each step runs a real comparison and marks itself done from your runs.
  An 85-term glossary with formulas, and theory pages for every algorithm.
- **Reproduce.** Describe runs and sweeps in YAML experiment files, commit them
  with your code, and rerun them anywhere.

## Labs

| Lab | What it studies | Examples |
| --- | --- | --- |
| Foundations | Tabular methods, bandits, exploration | Gridworld, Dyna Maze, CliffWalking, 10-armed bandit |
| Classic Control | Function approximation on small dynamics | CartPole, MountainCar, Acrobot, Pendulum |
| Continuous Control | Torque control of simulated bodies | 11 MuJoCo tasks, BipedalWalker |
| Perception | Acting from pixels | 5 MinAtar games |
| Multi-Agent | Two-player games | Tic-Tac-Toe, Connect Four, Texas Hold'em |
| Language models | RL fine-tuning of a tiny transformer | Verifiable-reward tasks, reward hacking |

Algorithms include Q-learning, SARSA, Dyna-Q, UCB, Thompson sampling, DQN (double,
dueling, prioritised, C51), REINFORCE, A2C, PPO, DDPG, TD3, SAC, CrossQ, PETS,
Dreamer-lite, GRPO (with Dr. GRPO and DAPO variants) and DPO. A diagnostics page
runs six probe environments with known answers against every algorithm.

## Quick start

Requires **Python 3.11** and **Node.js 20 or newer**.

```bash
git clone https://github.com/SauravBastola/rl-testbed.git
cd rl-testbed

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps            # installs the `rltestbed` command

cd web && npm install && npm run build && cd ..

rltestbed serve --port 8000           # open http://localhost:8000
```

Optional extras unlock more environments; the app works without them:

```bash
pip install "gymnasium[mujoco]"       # Continuous Control lab
pip install "gymnasium[box2d]"        # LunarLander, BipedalWalker
pip install rlcard                    # Texas Hold'em
```

From the terminal:

```bash
rltestbed train dqn cartpole --episodes 300
rltestbed run experiments/cartpole-baselines.yaml --dry-run
rltestbed export <run or sweep id> -o my-experiment.yaml
```

## Documentation

| | |
| --- | --- |
| [docs/CONTEXT.md](docs/CONTEXT.md) | Architecture, invariants and lessons learned: start here to contribute |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | Every algorithm, with its theory |
| [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) | Every environment |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Recent research and how it is reflected here, with sources |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Design decisions and why |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What is planned next |
| [docs/MUJOCO.md](docs/MUJOCO.md) | MuJoCo setup notes |

## Development

```bash
pip install -r requirements-dev.txt
make check          # lint, contrast checks, 145 backend + 50 frontend tests, build
cd web && npm run dev   # live-reloading UI on :5173 (run `rltestbed serve` alongside)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## Security

RL Testbed is a single-user tool with **no authentication**. It listens on
`127.0.0.1`, accepts only `localhost` host names (blocking DNS rebinding) and
grants no cross-origin access, so other websites cannot call it. Do not expose it
to a network you do not trust: anyone who can reach the port can start and delete
runs. Details in [docs/DECISIONS.md](docs/DECISIONS.md).

## Status

Actively developed; see the roadmap. Language models here are ~100k-parameter
character transformers: the algorithms are the real ones, the scale is not.

## License

[MIT](LICENSE) © Saurav Bastola
