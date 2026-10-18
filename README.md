# RL Testbed

A local laboratory for reinforcement learning. Pick an environment and an
algorithm, launch a run in an isolated process, watch the learning curve stream
in, compare configurations across seeds, keep the models worth keeping, and read
reference material about every method and environment, all in one web app with
a matching CLI.

The catalogue is organised into **labs**, one per shape of learning problem:

| Lab | What it studies | Environments |
|---|---|---|
| Foundations | Tabular methods on small, fully observable worlds | Gridworld, Dyna Maze, N-Chain, RiverSwim, Windy Gridworld, bandits, Catch, FrozenLake, CliffWalking, Taxi, Blackjack |
| Classic Control | Function approximation on low-dimensional dynamics | CartPole, MountainCar (+ continuous), Acrobot, Pendulum, LunarLander |
| Continuous Control | Torque control of simulated bodies | 11 MuJoCo tasks, BipedalWalker |
| Perception | Learning to act from pixels | 5 MinAtar games |
| Multi-Agent | Two-player games against an opponent | Tic-Tac-Toe, Connect Four, Texas Hold'em |
| Language models | RL fine-tuning of a tiny transformer with verifiable rewards | Addition, reversal, sorting, and a hackable reward |

Plus a global **Diagnostics** page: six probe environments with known answers
(five discrete, one continuous), shown as a pass/fail matrix across every algorithm.

**Catalogue:** 49 environments (43 for training, 6 diagnostic probes) and 37
algorithms (29 trainable, 8 planned, each with the reason it is not trainable
yet). Environments whose optional dependency is missing are listed as
unavailable with an install command, rather than hidden.

![Classic Control lab overview](docs/screenshots/lab_classic--midnight.png)

More in [docs/screenshots](docs/screenshots): the run composer, a live run, seed
comparison with a ±1σ band, a hyperparameter sweep, the diagnostics matrix, and
all three themes.

## Quick start

```bash
# Python 3.11 is required (gymnasium 0.29.1 and torch 2.4 are pinned against it)
python3.11 -m venv .venv && source activate.sh
pip install -r requirements.txt
pip install -e . --no-deps          # installs the `rltestbed` command

cd web && npm install && npm run build && cd ..   # build the web app once

rltestbed serve --port 8000         # open http://localhost:8000
```

Optional extras unlock more of the catalogue; the app runs without them:

```bash
pip install "gymnasium[mujoco]"     # Continuous Control lab
pip install "gymnasium[box2d]"      # LunarLander, BipedalWalker
pip install rlcard                  # Texas Hold'em
```

Your first run, from the browser: open a lab, press **New run**, pick an
environment and an algorithm (incompatible pairs are blocked with a reason before
anything starts), and launch. From the terminal:

```bash
rltestbed train dqn cartpole --episodes 300
rltestbed train sac half_cheetah --episodes 200 --keep

# Experiment files (the server must be running for `run`)
rltestbed run experiments/cartpole-baselines.yaml --dry-run
rltestbed run experiments/cartpole-baselines.yaml
rltestbed export <run, sweep or seed-group id>[,...] -o my-experiment.yaml
```

## Using the app

- **Lab overview:** the lab's newest run with its real curve, live counts, and
  a curriculum whose steps open pre-filled runs.
- **New run:** one page with three steps (choose, configure, review). Hyperparameter
  forms are generated from each algorithm's spec. Choose 3 or 5 seeds to launch
  them as one group.
- **Run detail:** live learning curve with greedy evaluations and the solved
  threshold, algorithm-specific metric charts, the episode table, configuration,
  downloadable artifacts, and the worker log. Stop, restart, clone, keep the model.
- **Sweeps:** pick parameters to vary (value lists, or uniform/log ranges for
  random search) and a number of seeds; every configuration runs through the
  queue. Results rank configurations by mean ± std, show each parameter's effect,
  and open the best configuration in New run with one click.
- **Experiments:** describe runs and sweeps in a YAML file, check it (every
  problem is reported with its location), and launch it. Any run, comparison or
  sweep exports back to such a file, so experiments can be committed, reviewed
  and rerun. See `experiments/cartpole-baselines.yaml`.
- **Compare:** seeds of the same configuration pool into a mean with a ±1σ band,
  with a summary (final, best eval, episodes to solve, wall clock), a per-seed
  table, and a list of exactly which settings differ.
- **Diagnostics:** "Run missing checks" queues every expected algorithm × probe
  pair; failed cells link to their runs.
- **Visualize:** step through one episode with a random policy or a kept model.
  Frames come from the same environment the agent acts in (MuJoCo renders in a
  short-lived subprocess so OpenGL works on macOS); builtin worlds without a
  renderer are drawn from their state; every step lists the observation, labelled
  where possible, and a note explains what is shown and why.
- `⌘K` / `Ctrl K` or `/` opens the command palette; `⌘⇧L` switches lab.
- **Language models:** GRPO, REINFORCE, DPO and supervised fine-tuning on a tiny
  pretrained transformer (no downloads, seconds per run). Visualize shows sampled
  answers shaded by token probability; a hackable task demonstrates reward hacking.
- **World models:** PETS plans with a learned ensemble at every step; Dreamer-lite
  trains an actor-critic inside the model. The run page's Imagination tab overlays
  the model's open-loop prediction on what really happened.
- **Learning path:** 9 modules, 23 experiments, from a random baseline to reading
  results across seeds. Each step teaches named concepts, runs in one click, asks
  a question, and marks itself done from your runs.
- **Glossary:** 85 terms by topic, with explanations, formulas, related terms and
  links to where each idea appears.
- **Night / Day** in the sidebar; **Settings** chooses each mode's theme
  (Night: Midnight, Graphite, Abyss, Obsidian. Day: Daylight, Paper, High contrast).

## Architecture

```
browser ──REST /api/*──► FastAPI (rltestbed/server/app.py) ──► SQLite registry
        ◄─WS /ws/runs/id─┘        │ queue (jobs.py, concurrency limit)
                                  ▼
                 subprocess: rltestbed.cli train-worker --run-id …
                 Trainer → history.jsonl, progress.json, checkpoints, worker.log
```

- **Specs are the single source of truth.** Each algorithm's `SPEC`
  (`build_spec` in `core/agents/base.py`) and each environment's `EnvSpec`
  generate the forms, documentation pages and compatibility rules. Algorithm
  theory lives in `core/agents/theory.py`.
- **Compatibility is decided in one place:** `capabilities.check_compat`, used
  by the API before launch and by every UI verdict.
- **Labs are data:** `rltestbed/labs.py`. A run belongs to the lab of its environment.
- **Runs are subprocesses.** A crash never takes down the server, and a run keeps
  training through a server restart. At most `RLTESTBED_MAX_CONCURRENT` runs
  train at once (default: half the CPUs, minimum 2); the rest wait as `queued`.
- **Seed aggregation is server-side:** `rltestbed/analysis.py`.

Frontend: React 18, Vite, Tailwind with design tokens (`web/src/styles.css`),
TanStack Query, Framer Motion, a custom SVG chart (`components/charts`), KaTeX
for algorithm theory, self-hosted fonts. See [docs/DECISIONS.md](docs/DECISIONS.md)
for the reasoning behind each choice.

## API additions since v3

All v2 endpoints and response shapes still work. New:

| Endpoint | Purpose |
|---|---|
| `GET /api/labs` | Lab definitions with their environments |
| `GET /api/system` | Versions, storage, installed extras, queue state |
| `POST /api/runs` with `seeds: [..]` | One run per seed under a shared `group_id` |
| `GET /api/runs?lab=&group_id=&status=&env=&agent=` | Filtered run list (each run has `lab`) |
| `PATCH /api/runs/{id}` | Edit name, notes, tags |
| `GET /api/runs/{id}/history?since=N` | Episodes after N, for cheap live polling |
| `GET /api/runs/{id}/artifacts/{path}` | Download an artifact |
| `GET /api/groups`, `GET /api/groups/{id}` | Seed groups and their aggregate curve |
| `GET /api/compare/aggregate?run_ids=&metric=&group_by=` | Mean ± std per configuration, plus config diff |
| `GET /api/diagnostics`, `POST /api/diagnostics/run` | Verdict matrix; queue probe runs |
| `POST /api/sweeps/preview` | Expand a search space: counts and errors, creates nothing |
| `POST /api/sweeps`, `GET /api/sweeps?lab=` | Launch a sweep; list sweeps with progress |
| `GET /api/sweeps/{id}?rank_by=final\|best_eval` | Ranked configurations and per-parameter effects |
| `POST /api/sweeps/{id}/stop`, `DELETE /api/sweeps/{id}` | Stop or delete a sweep and its runs |
| `POST /api/experiments/validate`, `POST /api/experiments/launch` | Check or launch an experiment file (`{"text": "..."}`) |
| `GET /api/experiments/export?run_ids=&sweep_id=` | Experiment file that recreates runs and/or a sweep |
| `GET /api/experiments/examples` | One starter file per lab curriculum |
| `GET /api/curriculum` | Learning-path modules with each step's live status and experiment file |
| `GET /api/runs/{id}/history.csv`, `GET /api/compare/aggregate?...&format=csv` | CSV exports |

Environments gain `lab`, `available`, `unavailable_reason`, `requires`,
`sparse_reward` and `reward_threshold`. Algorithms gain `status_reason`, `paper`
and `docs.math`.

## Extending

- **Algorithm:** subclass `Agent`, set `SPEC = build_spec(...)`, register it in
  `core/agents/__init__.py`, add theory to `theory.py`, and add it to a probe's
  `should_pass` in `benchmarks.py`. Its form, documentation page and
  compatibility checks follow automatically.
- **Environment:** drop a file into `custom_envs/` (see its README), or register
  it in `core/envs/`. Set `requires` for optional dependencies and `lab` to place
  it in a specific lab.
- **Lab:** append a dict to `LABS` in `rltestbed/labs.py`.

## Development

```bash
pip install -r requirements-dev.txt  # adds pytest
make help                            # list common tasks; `make check` runs what CI runs
python -m pytest tests -q            # 145 tests
cd web
npm run dev                          # :5173, proxies /api and /ws to :8000
npm run lint                         # ESLint, including a no-hex-colour rule
npm run test                         # Vitest: every route plus helpers (50 tests)
npm run check:tokens                 # hex scan + WCAG contrast for all themes
npm run format                       # Prettier
npm run build                        # the server serves web/dist
```

## Upgrade note (v3.7): re-run continuous-control results

Before v3.7, SAC, TD3 and DDPG stored observations in their replay buffers
normalised by the statistics of the moment they were stored, which left early
samples inconsistent with later ones. On Pendulum, SAC did not learn within 40
episodes; with the fix it reaches about -130. Results from these algorithms
made with earlier versions understated them and are worth re-running.

## Security

RL Testbed is a single-user local tool with **no authentication**.

- By default it listens on `127.0.0.1` and only answers requests addressed to
  `localhost` / `127.0.0.1`, which blocks DNS-rebinding attacks from websites.
- No cross-origin access is granted, so other websites open in your browser
  cannot call the API. The Vite dev server proxies `/api`, so development needs
  no CORS. To allow a specific origin: `RLTESTBED_CORS_ORIGINS=http://host:port`.
- `rltestbed serve --host 0.0.0.0` makes it reachable from your network and
  prints a warning: anyone who can reach the port can start, stop and delete
  runs. Restrict accepted host names with `RLTESTBED_ALLOWED_HOSTS=a.local,10.0.0.5`.

## Notes

- `rltestbed/compat.py` restores `np.float_` for Gymnasium 0.29.1's Acrobot under
  NumPy 2 and strips an attribute MuJoCo ≥ 3 rejects from bundled XML assets.
- MuJoCo rendering picks an OpenGL backend per platform (GLFW on macOS and
  desktops, EGL on headless Linux). Set `MUJOCO_GL` to override.
- Pixel observations are flattened for the MLP networks; the compatibility check
  warns about this on every pixel environment.
- In two-player games the learner plays one seat against a random legal opponent.
  Self-play is planned.
- Stopping a run finishes the current episode and writes a checkpoint; stopped,
  failed and interrupted runs can be restarted.
- Contributing and the patch workflow: [CONTRIBUTING.md](CONTRIBUTING.md).
  What is planned next: [docs/ROADMAP.md](docs/ROADMAP.md).
- Where RL is heading and what to add next, with sources: [docs/RESEARCH.md](docs/RESEARCH.md).
- Further reference: [docs/ALGORITHMS.md](docs/ALGORITHMS.md),
  [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md), [docs/MUJOCO.md](docs/MUJOCO.md).
