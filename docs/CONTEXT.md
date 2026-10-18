# Project context

Everything someone (or an AI assistant) needs to continue work on RL Testbed
without reading the whole history. Last updated for v3.7, October 2026.

## What this is

A local, single-user laboratory for reinforcement learning: a Python backend
(FastAPI, SQLite, one subprocess per training run) and a React web app, plus a
CLI. Every algorithm is implemented in this repository, readably, so it can be
studied as well as run. It doubles as a learning resource: a 23-step learning
path, an 85-term glossary, and theory pages for every algorithm.

## Layout

```
rltestbed/                 Python package
  core/agents/             one module per algorithm family; registry in __init__.py
    base.py                Agent interface, ParamSpec, build_spec (metadata for the UI)
    continuous.py          SAC, TD3, DDPG, networks (MLP and SimBa), replay buffer
    crossq.py              CrossQ (SAC without target networks)
    world_models.py        EnsembleModel, PETS, Dreamer-lite
    theory.py              LaTeX theory text shown on algorithm pages
  core/envs/               environment registry, adapters, renderers, observation labels
    builtin/               tabular and classic environments written here (with view())
  core/lm/                 language-model RL: TinyGPT, tasks, base models, GRPO/DPO, SequenceTrainer
  core/trainer.py          step-by-step training loop (writes progress.json, history.jsonl)
  core/runner.py           create_trainer(): the only branch between the two training loops
  launch.py                the single place runs and sweeps are validated and created
  experiments.py           YAML experiment files: parse, launch, export
  sweeps.py, analysis.py   sweep planning; seed aggregation
  rollout.py               Visualize episodes (MuJoCo renders in a subprocess)
  curriculum.py, glossary.py, labs.py   teaching content, all data
  server/app.py            REST API and static hosting of web/dist
  registry.py, jobs.py     SQLite registry; run queue and worker subprocesses
  cli.py                   `rltestbed` command (serve, train, run, export, workers)
web/                       React 18 + Vite + Tailwind + TanStack Query
tests/                     pytest (145 tests); web/src/test (Vitest, 50 tests)
docs/                      reference and decision records
experiments/               example experiment files
```

## Invariants: keep these true

1. **One launch path.** The API, experiment files and the CLI all create runs
   through `launch.py`. Never validate or create runs anywhere else.
2. **Runs are subprocesses** writing `progress.json` and `history.jsonl` under
   `runs/<id>/`. Any new training loop must write the same files (see
   `core/lm/trainer.py`); then charts, Compare, sweeps and the learning path work
   unchanged.
3. **Compatibility is declared, not special-cased.** Agents list the observation
   and action types they support; environments declare theirs; `capabilities.py`
   decides. Language tasks use `tokens`, which is why DQN cannot run on them.
4. **Teaching content is data.** Glossary entries, curriculum steps, labs and
   theory live in Python data structures. `tests/test_learning.py` fails if any
   cross-reference stops resolving or a step's experiment stops validating.
5. **No hex colours in components.** Colours are tokens; `npm run check:tokens`
   checks contrast for all seven themes and that every lab has an accent.
6. **Measure before claiming.** Every algorithm's description and the docs report
   measured behaviour, including "no clear gain at this scale".

## Lessons from bugs (read before touching the related code)

- **Normalise replay data when sampling, not when storing.** Until v3.7 the
  off-policy agents stored normalised observations; early samples kept stale
  statistics and SAC never learned Pendulum. Regression test in `test_modern.py`.
- **Truncation is not termination.** The trainer sets `agent.last_step_truncated`;
  bootstrapping agents and world models must use it. Pendulum only ever truncates.
- **Shapes in TD targets.** SAC once regressed every Q-value toward the batch mean
  because a `(B, 1)` log-prob broadcast against `(B,)` values.
- **Tests must reach the code they test.** An HTTP client normalised `../` before
  the request was sent, so a path-traversal test passed without testing anything.
  Encode such paths (`..%2F`).
- **Exploits must be reachable to appear.** Reward hacking in the language lab only
  showed once the base model sometimes produced long answers.
- **Pretrain to convergence for reproducible baselines.** Stopping mid-way gave
  4-31% base accuracy across machines.
- **macOS OpenGL needs the main thread.** MuJoCo renders in `rollout-worker`.
- **Node 22+ ships its own `localStorage`**, which shadows jsdom's in tests; the
  test setup installs an in-memory one.

## Workflow

- Changes are delivered as `git format-patch` series and applied with
  `git am --3way`. Each patch builds on the previous one: check that
  `git log --oneline -1` shows the previous patch's last commit before applying.
- `make check` runs lint, token checks, backend and frontend tests and the build.
  It uses `.venv/bin/python` automatically when that exists.
- Commits explain why, with measurements where behaviour changed.
- `CONTRIBUTING.md` has the full patch workflow.

## Where to look

| Task | Start in |
| --- | --- |
| Add an algorithm | `core/agents/` (copy a sibling), register in `core/agents/__init__.py`, add theory in `theory.py`, a lab in `labs.py` |
| Add an environment | `core/envs/__init__.py` (`_register`), or drop a file in `custom_envs/` |
| Add a learning-path step | `curriculum.py` (experiment-file format) |
| Add a glossary term | `glossary.py` |
| Change what Visualize shows | `rollout.py`, `core/envs/observations.py` |
| Add an API endpoint | `server/app.py`; keep existing response shapes (additive changes only) |
| Change the UI | `web/src/pages`, components in `web/src/components` |

## Status and next steps

Done through v3.7: six labs (tabular, classic control, MuJoCo, MinAtar, games,
language models), 29 trainable algorithms including PETS, Dreamer-lite, CrossQ,
GRPO/DAPO and DPO, sweeps, experiment files, diagnostics, a learning path.

Planned, in suggested order (details in `docs/ROADMAP.md`):

1. A dedicated **World models** lab, so PETS and Dreamer-lite are easier to find.
2. **Resume from checkpoint**, then **sweep early stopping** (successive halving).
3. **Offline RL**: record datasets from runs; behaviour cloning, CQL.
4. A **TD-MPC-style** agent (latent planning with a learned terminal value).
5. **Self-play** for the board games.

Known limitations: single user, no authentication (keep it on localhost);
language models are ~100k parameters (the algorithms are real, the scale is
not); MuJoCo rendering is slow on CPUs without GPU OpenGL.
