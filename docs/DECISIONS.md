# Design decisions (v3)

This records what was decided during the v3 rebuild and why, including where the
handoff brief was followed, where it was changed, and what the audit got wrong.

## Information architecture: five labs, diagnostics is global

**Decision.** Five labs, defined by the shape of the learning problem:
Foundations, Classic Control, Continuous Control, Perception and Multi-Agent.

**Deviation from the brief.** The brief proposed a sixth "Diagnostics" lab. A lab
holding only the five probe environments would be a page pretending to be a
workspace; the brief itself says not to add a lab that is only a rename of another
surface. Diagnostics is therefore a global page, and probes belong to no lab.

**Labs are data, in the backend.** `rltestbed/labs.py` holds the definitions
(label, tagline, accent, glyph, membership rule, algorithm priority, metric
palette, curriculum) and `lab_for_env()` decides membership. The frontend reads
`GET /api/labs`; it never hard-codes a lab. Adding a lab is one dict. An
environment can override its lab with `EnvSpec.lab` or `ENV_SPEC["lab"]`.

**A run's lab is derived, not stored.** A run belongs to the lab of its
environment. This replaces the brief's `ALTER TABLE runs ADD COLUMN lab`: a
stored copy could drift from the catalogue, a derived one cannot.

**Boundaries.** Hard for the catalogue (each lab shows its own environments, with an
"All labs" switch); soft for runs (lab-filtered by default, "All labs" one click
away; Compare, Models, Visualize and Diagnostics are global).

## Routing

`/lab/:labId`, `/lab/:labId/new`, `/lab/:labId/runs`, `/lab/:labId/envs`,
`/lab/:labId/algorithms` are lab-scoped. Entity pages (`/runs/:id`, `/envs/:name`,
`/algorithms/:name`) are global and adopt their entity's lab accent. All v2 URLs
(`/train`, `/new`, `/runs`, `/envs`, `/algorithms`, `/benchmarks`) redirect.

## Themes: three peers

Midnight (default dark), Daylight (true light theme with darker chart hues) and
Graphite (low-chroma neutral for long sessions). Aurora and Ember were hue
rotations and are retired; a stored retired theme falls back to Midnight.

The v2 `!important` daylight bridge is deleted, not extended: every component uses
semantic tokens, so nothing needs repainting per theme. `npm run check:tokens`
computes WCAG contrast for 23 colour pairs in every theme from `styles.css` itself
and fails the check if any pair is below its minimum (4.5:1 for text and status
colours, 3:1 for chart lines and accents).

## Tokens and type

- Colours are RGB-triplet custom properties consumed by Tailwind. Three layers:
  primitive hues (`--hue-*`, re-tuned per theme), semantic (`--bg`, `--text`…) and
  the lab accent (`--accent`, re-pointed by `[data-lab]` on `<html>`).
- No hex literal in any component. Enforced twice: an ESLint rule and the token check.
- Type scale `micro 11 / caption 12 / body-s 13 / body 14 / title-s 16 / title 20 /
  display 30 / hero 40`. Tailwind's default font sizes are removed from the config,
  so an off-scale size cannot be written by accident.
- Radii `6 / 8 / 12 / 16 / full`; three elevations (flat, raised, overlay); motion
  `120 / 200 / 320 ms` with one easing.
- Fonts are self-hosted via Fontsource. The app makes no third-party requests.

## Visual language

Kept from Prism: the cool dark canvas, Space Grotesk / Inter / JetBrains Mono,
the ambient wash and fading grid. Changed: sentence-case labels instead of
tracked uppercase eyebrows, no gradient text, monospace only for compared numbers,
one card component in two paddings. The memorable element is the per-lab
"specimen": a hand-drawn SVG of the problem each lab studies (a gridworld policy,
a phase portrait, joint torques, a pixel frame, a game board). Everything else stays
quiet. The "live" label on a static illustration is gone; the hero curve on a lab
overview is always a real run.

## State and data

TanStack Query for every fetch (cache, deduplication, polling only while something
is live). The run WebSocket updates the cached run; history is polled with the new
`?since=` parameter so a live chart never re-downloads its whole curve. Theme and
current lab live in one small React context. No global store.

## Charts

A custom SVG `LineChart` inside `ChartFrame` replaces Recharts. Reasons: ±1σ bands,
the solved threshold, faint raw-behind-smoothed lines and token colours are all
first-class, and the bundle is ~100 KB lighter. `ChartFrame` owns the title,
legend, metric switcher, smoothing control, PNG export (theme colours are resolved
before export) and the empty/loading states. Charts draw once on mount and never
animate on update. A threshold far outside the data range is labelled instead of
being allowed to flatten the curve.

## Seeds and comparison

`POST /api/runs` accepts `seeds: [...]` and creates one run per seed under a shared
`group_id` (new column, migrated in place). The response keeps `run_id` and adds
`run_ids` and `group_id`. Aggregation is server-side (`rltestbed/analysis.py`),
used by `GET /api/groups/{id}` and `GET /api/compare/aggregate`. Runs are pooled by
configuration signature (agent, environment, hyperparameters) by default, so seeds
launched separately still pool. The Compare page warns when every configuration has
a single seed, and lists exactly which settings differ.

## Job queue

`JobManager` now has a concurrency limit (`RLTESTBED_MAX_CONCURRENT`, default
half the CPU count, minimum 2). Extra runs wait as `queued` and a server task starts
them as slots free up. Without this, "run all diagnostics" or a 5-seed launch would
start dozens of processes at once. Runs remain subprocesses and survive a server
restart (verified by killing the server mid-run).

## Diagnostics

Each benchmark has a `pass_threshold`, so the matrix shows real verdicts. Probe
runs use small update intervals (`DIAGNOSTIC_OVERRIDES` in `benchmarks.py`): one-step
episodes give only a few hundred transitions, too few to trigger a PPO update or a
DQN target sync at the CartPole-tuned defaults. Without the overrides, those
failures pointed at the budget, not at a bug.

## Coverage added

Algorithms (13 → 21 trainable): Expected SARSA, Double Q-learning, Dyna-Q, Monte
Carlo control, UCB, Thompson sampling, DQN with prioritised replay, C51. Every new
algorithm passes at least one probe (tests in `tests/test_v3.py`). Nine planned
algorithms each carry a `status_reason`.

Environments: Dyna Maze, Drifting Bandit, Connect Four. Environments whose optional
dependency is missing (MuJoCo, Box2D, RLCard) are listed with `available: false` and
an install hint instead of disappearing.

## Bugs found that the audit missed

1. Gymnasium registration dropped `action_type`: Pendulum and BipedalWalker were
   "discrete", so DQN was allowed and SAC/TD3/DDPG were blocked.
2. No learning algorithm accepted `image` observations, so all of MinAtar (the
   proposed Perception lab) could only run the random agent.
3. The PettingZoo adapter summed both seats' rewards in zero-sum games, so the
   learning signal was identically zero. The learner now plays one seat against a
   random legal opponent.
4. A missing MuJoCo silently removed the whole suite from the catalogue.
5. Tabular agents kept exploring during greedy evaluation; policy-gradient agents
   sampled during it. Both made "best eval" noisy. Evaluation is now greedy everywhere.
6. The random agent guessed "continuous" from the observation length (3 or 24).
7. `AlgorithmDetail.jsx` carried a second, hand-written copy of algorithm theory.
   It now lives in `rltestbed/core/agents/theory.py`, read by `build_spec`.
8. A worker that failed during construction left its run "running" forever.
9. A2C gathered log-probabilities with `torch.arange(n)` (the timestep index)
   instead of the actions taken, so it reinforced arbitrary actions; its entropy
   term was computed from those gathered values. Found by the Diagnostics matrix.
10. A2C and REINFORCE normalised advantages within each episode. On short episodes
    that maps any two advantages to about ±0.7 and discards the value baseline.
    Normalisation now needs at least 8 steps.
11. (v3.1) SAC's actor returned log-probabilities shaped (B, 1) while the critics
    return (B,). The critic target broadcast to (B, B), regressing every Q-value
    toward the batch-average target. Found when the new continuous probe ran
    with warnings visible.

The audit was also wrong that runs had no tags: `tags_json` already existed.

12. (v3.1.1) The frontend catch-all route served `dist / full_path` without
    checking that the result stayed inside `dist`. A percent-encoded request such
    as `/..%2F..%2Fetc%2Fpasswd` returned any file the user could read. Combined
    with CORS allowing every origin, any website open in the browser could read
    local files while the server ran. Both predate v3; found because a test meant
    to check traversal never reached its handler (HTTP clients normalise `../`).

## v3.1.1: local security model

No authentication, so the defences are about who can reach the server:
loopback binding, a Host-header allowlist (DNS rebinding), no cross-origin
access unless configured, and file serving confined to its directory. Startup
and shutdown hooks moved from the deprecated `on_event` to a lifespan handler.

## v3.1: continuous diagnostics and Gaussian PPO

Every probe was discrete, so SAC, TD3 and DDPG never appeared in the Diagnostics
matrix. Probe 6 is a one-step continuous task (reward 1 - |a - 0.5|) that all four
continuous learners must solve. Gaussian PPO subclasses PPO and overrides only the
`_evaluate` hook, the network and action sampling, so the GAE and clipping logic
exists once.

## v3.1: hyperparameter sweeps

**A configuration is a seed group.** Each configuration in a sweep gets its own
`group_id`, so pooling seeds into mean ± std reuses `analysis.aggregate` instead of
a second implementation. Runs also carry `sweep_id`; a sweep's status is derived
from its runs rather than stored, for the same reason a run's lab is derived.

**Validation happens before creation.** `sweeps.plan` refuses invalid spaces and
anything over 200 runs (configurations × seeds), checking the grid size before
expanding it. Every configuration passes the same `validate_params` as a single
run. `POST /api/sweeps/preview` returns errors as data, so the composer can show
them live without catching exceptions.

**Ranking is by mean final return by default**, switchable to best evaluation.
Final return (last 10% of training episodes) reflects the whole learning process;
best evaluation rewards a single lucky greedy check. The detail page states the
seed count next to the winner and warns when it is one.

**No early stopping yet.** Successive halving would save compute but needs runs to
be resumable mid-way; it is on the roadmap behind "resume from checkpoint".

## v3.2: experiment files

**YAML, strict.** YAML because it is what people already write configs in and it
diffs well in git. Every entry is validated by a pydantic model with
`extra="forbid"`, so `episods: 300` is an error rather than a silently ignored key,
and errors carry a location (`runs[0].episods`, or a line and column for syntax
errors).

**One launch path.** `rltestbed/launch.py` is the only code that validates and
creates runs and sweeps. The API endpoints, experiment files and the CLI all call
it, so a file can never do something the New run page would refuse.

**All or nothing.** A file is validated completely before the first run is
created. Half-launched experiments are worse than none: they look like results.

**Exports are explicit.** Exported runs list every hyperparameter, not only those
that differ from the defaults, so a committed file keeps its meaning if a default
changes later. Runs that differ only by seed collapse into one entry.

**The CLI talks to the server for launches.** `rltestbed run` validates locally,
then posts to the running server, so its queue schedules the work and the web
app shows it live. `rltestbed export` reads the local registry directly and needs
no server.

## v3.3: visualisation and themes

**Pictures come from the acting environment.** v3 stepped a second "twin"
environment for frames, reset with a different seed, so pictures could show a
different episode than the numbers. Adapters now accept a render mode and the
rollout renders the instance the agent acts in. Each step pairs the state the
agent acted in with its observation; a final record shows the end state.

**MuJoCo renders in a subprocess.** macOS's GLFW must run on a main thread and
request handlers are not; a short-lived `rollout-worker` process sidesteps that
on every platform and isolates native crashes. The backend is chosen per
platform and never set in the server process.

**Draw state, do not screenshot text.** Builtin worlds expose `view()` data and
the web app draws them as SVG, in the current theme. Where there is genuinely
nothing to draw (probes), the page says so and explains the observation.

**Two modes, several themes.** Most people switch between dark and light, not
between seven palettes, so the sidebar offers Night and Day and Settings picks
each mode's theme. Theme CSS is scoped to `[data-theme]` so live previews work;
every theme is checked for contrast.

## Component conventions

The conventions asked for one component per file. Primitives are instead grouped by
family (`ui/Field.jsx` exports `Input`, `Select`, `Switch`…), because a dozen
three-line files for one form family is harder to navigate, not easier. Pages are
one component per file.

## Deliberately not done

- Offline RL, self-play, independent multi-agent learners, NoisyNet, Rainbow, TRPO
  and Gaussian PPO are planned with reasons, not implemented.
- No Atari/Procgen/MiniGrid environments were added; `availability.py` already
  knows the `minigrid` extra if someone registers tasks for it.
- No in-browser TensorBoard; artifacts are listed and downloadable.
