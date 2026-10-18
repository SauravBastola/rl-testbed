"""RL Testbed - a comprehensive reinforcement learning experimentation platform.

Provides:
- A unified registry of agents and environments
- A trainer with TensorBoard logging, evaluation and checkpointing
- A metadata registry (SQLite) for runs and models
- Both a CLI (typer) and a web app (FastAPI + React)
"""

__version__ = "2.0.0"

# Restore compatibility with newer numpy / mujoco releases before any
# environment or agent module touches them. Safe to call repeatedly.
from rltestbed.compat import apply_compat_shims  # noqa: E402

apply_compat_shims()
