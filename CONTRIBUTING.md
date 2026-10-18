# Working on RL Testbed

## Branches and commits

`main` is always runnable: tests, lint and the build pass on every commit. Work on
a branch, keep each commit focused on one change, and write the subject line in
the imperative ("Add sweep API", not "Added sweep API").

## Applying a patch series

Changes arrive as a patch file produced by `git format-patch`. Each file contains
one or more commits with their messages. To apply one:

```bash
git switch main
git pull                       # if you use a remote
git switch -c <topic>          # e.g. v3.1-sweeps
git am --3way path/to/series.patch
make check                     # or the commands listed in the README
git switch main && git merge --ff-only <topic>
```

If `git am` stops on a conflict: fix the files, `git add` them, then
`git am --continue`. To give up and return to where you were: `git am --abort`.

To produce a patch series yourself (for review or to share):

```bash
git format-patch main --stdout > my-change.patch
```

## Before you commit

```bash
make check
```

This runs the Python tests, ESLint, the Prettier check, the design-token and
contrast check, the frontend tests and the production build: the same checks as
CI (`.github/workflows/ci.yml`).

## Where things go

- New algorithm: `rltestbed/core/agents/`, registered in `__init__.py`, with
  theory in `theory.py` and an entry in a probe's `should_pass` (`benchmarks.py`).
- New environment: `custom_envs/` (no code change needed) or `rltestbed/core/envs/`.
- New lab: one dict in `rltestbed/labs.py`.
- Design decisions: add a section to `docs/DECISIONS.md`.
- Roadmap changes: `docs/ROADMAP.md`.
