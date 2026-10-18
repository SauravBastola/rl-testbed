# CLAUDE.md — Working conventions

Operational rules for this repository. Read `01-BRIEF.md` for the mission; this file is the how.

## Repository layout

```
rltestbed/     Python package (backend). Stable — extend, don't rewrite.
web/           React frontend (Vite). This is where most work happens.
docs/          Reference docs. Keep in sync with the code.
custom_envs/   Drop-in user environments, auto-discovered on restart.
tests/         pytest suite. Must stay green.
runs/          Run artifacts (generated). Never commit.
models/        Promoted checkpoints (generated). Never commit.
data/          SQLite registry (generated). Never commit.
```

## Commands

```bash
# Backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m rltestbed.cli serve --port 8000
python -m pytest tests -q

# Frontend
cd web
npm install
npm run dev      # :5173, proxies /api and /ws to :8000
npm run build    # required for the server to serve new UI
npm run lint
npm run test
npm run check:tokens   # hex scan + contrast per theme
```

**Always build before verifying through the server.** The server serves `web/dist`; editing `web/src` changes nothing until you rebuild.

## Style

**Python**
- `black`, line length 88, target py311. `legacy/` is excluded.
- Type hints on public functions. No `Any` where a real type exists.
- Metadata belongs in `SPEC` / `EnvSpec`, never in a second location.

**JavaScript / React**
- Function components, hooks, no class components.
- One component per file; file name matches the default export.
- No hex literals in components — use tokens (`bg-surface`, `text-muted`, `border-subtle`).
- No raw pixel font sizes — use the type scale.
- Tailwind for layout and spacing; CSS custom properties for anything theme-dependent.
- `cn()` from `lib/utils.js` for conditional classes.
- Data fetching through the query layer, not `useEffect` + `useState` per page.

## Invariants

1. `capabilities.check_compat` is the only place compatibility is decided.
2. Agent/environment documentation is generated from `SPEC` / `EnvSpec`.
3. Training runs in a subprocess; the server must survive a worker crash.
4. Existing CLI commands and REST response shapes keep working.
5. Python stays `>=3.11,<3.12`.
6. Missing optional dependencies degrade gracefully; the app always boots.

## Adding an algorithm

1. Subclass `Agent` in `rltestbed/core/agents/`.
2. Implement `get_action`, `update`, `get_metrics`, `state_dict`, `load_state_dict`.
3. Define `SPEC = build_spec(...)` with full encyclopedia fields.
4. Register in `core/agents/__init__.py`.
5. Verify on a diagnostic probe; add it to the relevant benchmark's `should_pass`.

## Adding an environment

1. Implement `reset()` / `step()` (Gymnasium-style 5-tuple, or builtin dict-style).
2. Export `ENV_SPEC` and `make()`.
3. Register in `core/envs/suites.py`, or drop the file in `custom_envs/`.
4. Fill the descriptive fields — `reward_desc`, `recommended_agents`, `limitations`, exercises. They are the catalogue.

## Definition of done for a change

- Tests pass; new backend behaviour has a test.
- `npm run build` succeeds with no new warnings.
- The affected route renders at 768px and 1440px.
- All four states (loading, empty, error, content) are handled.
- No console errors.
- Docs updated if behaviour or counts changed.

## Do not

- Do not rewrite `registry.py`, `jobs.py`, or `trainer.py` wholesale.
- Do not add a UI component library that replaces the token system.
- Do not duplicate compatibility rules, algorithm copy, or environment copy.
- Do not commit `runs/`, `models/`, `data/`, `web/dist/`, `node_modules/`, or `__pycache__/`.
- Do not leave `"Loading…"` strings or unlabelled icon buttons.
- Do not present a single-seed result as a comparison.

## Labs (v3)

Labs are defined in `rltestbed/labs.py` and served by `GET /api/labs`. A run's lab
is derived from its environment. The frontend never hard-codes lab membership.
See `docs/DECISIONS.md` for the reasoning behind every v3 choice.
