"""Central path configuration for the testbed.

All artifacts live under the project root (or RLTESTBED_HOME if set):
    runs/<run_id>/          per-run directory (config, progress, tb logs, checkpoints)
    models/                 promoted ("kept") models
    data/testbed.db         SQLite metadata registry
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("RLTESTBED_HOME", Path(__file__).resolve().parent.parent)
)

RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_ENVS_DIR = PROJECT_ROOT / "custom_envs"
WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"

DB_PATH = DATA_DIR / "testbed.db"

for d in (RUNS_DIR, MODELS_DIR, DATA_DIR, CUSTOM_ENVS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id
