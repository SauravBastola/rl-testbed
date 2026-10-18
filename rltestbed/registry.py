"""SQLite-backed metadata registry for runs and models with schema migrations.

Every run records:
  - 8-char random unique id
  - Friendly name, tags, description/notes
  - Algorithm, environment, seed, episodes
  - Git commit, platform/host info, device info
  - Parent run (for resumed or cloned runs)
  - Full parameter and config snapshots
  - Timestamps and live status

Kept models record:
  - Unique model ID (8-char lowercase alphabets only, 12 if needed)
  - Checkpoint type (best | final | manual | imported)
  - Checksum (SHA-256), file size, architecture metadata
  - Favorite and archived flags, user tags and notes
"""

import hashlib
import json
import os
import platform
import secrets
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rltestbed.config import DB_PATH, MODELS_DIR

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent TEXT NOT NULL,
    env TEXT NOT NULL,
    seed INTEGER NOT NULL,
    episodes INTEGER NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    config_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'created',
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    episodes_done INTEGER DEFAULT 0,
    best_eval_reward REAL,
    final_mean_reward REAL,
    pid INTEGER,
    notes TEXT DEFAULT '',
    model_kept INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    run_id TEXT,
    agent TEXT NOT NULL,
    env TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER DEFAULT 0,
    best_eval_reward REAL,
    final_mean_reward REAL,
    metrics_json TEXT DEFAULT '{}',
    description TEXT DEFAULT '',
    created_at REAL NOT NULL
);
"""

# Migration columns added in V2
_MIGRATIONS_RUNS = [
    ("tags_json", "TEXT DEFAULT '[]'"),
    ("parent_run_id", "TEXT"),
    ("git_commit", "TEXT DEFAULT ''"),
    ("host_info", "TEXT DEFAULT ''"),
    ("device", "TEXT DEFAULT 'cpu'"),
    # v3: runs launched together with different seeds share a group id
    ("group_id", "TEXT"),
]

_MIGRATIONS_MODELS = [
    ("tags_json", "TEXT DEFAULT '[]'"),
    ("checkpoint_type", "TEXT DEFAULT 'best'"),  # best | final | manual | imported
    ("checksum", "TEXT DEFAULT ''"),
    ("version", "TEXT DEFAULT '1.0'"),
    ("is_favorite", "INTEGER DEFAULT 0"),
    ("is_archived", "INTEGER DEFAULT 0"),
    ("architecture_json", "TEXT DEFAULT '{}'"),
]


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.executescript(_SCHEMA_V1)
        # Apply columns safely if upgrading from existing DB
        for table, cols in [("runs", _MIGRATIONS_RUNS), ("models", _MIGRATIONS_MODELS)]:
            existing_cols = {
                r["name"] for r in con.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col_name, col_type in cols:
                if col_name not in existing_cols:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


init_db()


def new_id() -> str:
    """8-char lowercase alphanumeric unique id (runs)."""
    for _ in range(100):
        rid = "".join(
            secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8)
        )
        with _conn() as con:
            exists_runs = con.execute("SELECT 1 FROM runs WHERE id=?", (rid,)).fetchone()
            exists_models = con.execute("SELECT 1 FROM models WHERE id=?", (rid,)).fetchone()
        if not exists_runs and not exists_models:
            return rid
    raise RuntimeError("could not generate unique id")


# Model IDs: 8-char alphabets only (lowercase a-z). Change length to 12 if needed.
MODEL_ID_LENGTH = 8
MODEL_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def new_model_id(length: int = MODEL_ID_LENGTH) -> str:
    """8-char (or 12) lowercase alphabets only, unique across runs + models."""
    for _ in range(100):
        mid = "".join(secrets.choice(MODEL_ID_ALPHABET) for _ in range(length))
        with _conn() as con:
            exists_runs = con.execute("SELECT 1 FROM runs WHERE id=?", (mid,)).fetchone()
            exists_models = con.execute("SELECT 1 FROM models WHERE id=?", (mid,)).fetchone()
        if not exists_runs and not exists_models:
            return mid
    raise RuntimeError("could not generate unique model id")


def now() -> float:
    return time.time()


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT", "")


def get_host_info() -> str:
    try:
        return f"{platform.node()} ({platform.system()} {platform.release()})"
    except Exception:
        return "unknown"


def compute_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ runs --
_RUN_COLS = (
    "id", "name", "agent", "env", "seed", "episodes",
    "params_json", "config_json", "status", "created_at",
    "started_at", "finished_at", "episodes_done",
    "best_eval_reward", "final_mean_reward", "pid", "notes",
    "model_kept", "tags_json", "parent_run_id", "git_commit",
    "host_info", "device", "group_id"
)


def create_run(
    name: str,
    agent: str,
    env: str,
    seed: int,
    episodes: int,
    params: Dict,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    notes: str = "",
    parent_run_id: Optional[str] = None,
    device: str = "cpu",
    group_id: Optional[str] = None,
) -> str:
    rid = new_id()
    git_commit = get_git_commit()
    host_info = get_host_info()

    with _conn() as con:
        con.execute(
            """
            INSERT INTO runs (
                id, name, agent, env, seed, episodes, params_json, config_json,
                status, created_at, notes, tags_json, parent_run_id, git_commit,
                host_info, device, group_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rid,
                name,
                agent,
                env,
                seed,
                episodes,
                json.dumps(params),
                json.dumps(config or {}),
                "created",
                now(),
                notes,
                json.dumps(tags or []),
                parent_run_id,
                git_commit,
                host_info,
                device,
                group_id,
            ),
        )
    return rid


def update_run(run_id: str, **fields):
    if not fields:
        return
    cols, vals = [], []
    for k, v in fields.items():
        if k not in _RUN_COLS:
            # support direct dict fields
            if k in ("params", "config", "tags"):
                k_json = f"{k}_json"
                cols.append(f"{k_json}=?")
                vals.append(json.dumps(v))
                continue
            raise ValueError(f"unknown run column {k}")
        if k in ("params_json", "config_json", "tags_json") and isinstance(v, (dict, list)):
            v = json.dumps(v)
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(run_id)
    with _conn() as con:
        con.execute(f"UPDATE runs SET {', '.join(cols)} WHERE id=?", vals)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_runs(limit: int = 500, agent: Optional[str] = None, env: Optional[str] = None,
              status: Optional[str] = None, group_id: Optional[str] = None,
              envs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM runs WHERE 1=1"
    params: List[Any] = []
    if group_id:
        query += " AND group_id=?"
        params.append(group_id)
    if envs is not None:
        if not envs:
            return []
        query += f" AND env IN ({','.join('?' * len(envs))})"
        params.extend(envs)
    if agent:
        query += " AND agent=?"
        params.append(agent)
    if env:
        query += " AND env=?"
        params.append(env)
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _conn() as con:
        rows = con.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def new_group_id() -> str:
    return "g" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(7))


def list_groups(limit: int = 200) -> List[Dict[str, Any]]:
    """One row per run group with its member count and status mix."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT group_id, MIN(name) AS name, MIN(agent) AS agent, MIN(env) AS env,
                   COUNT(*) AS n_runs, MIN(created_at) AS created_at,
                   SUM(status='completed') AS completed,
                   SUM(status IN ('running','stopping')) AS running,
                   AVG(final_mean_reward) AS mean_final_reward
            FROM runs WHERE group_id IS NOT NULL
            GROUP BY group_id ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_run(run_id: str):
    with _conn() as con:
        con.execute("DELETE FROM runs WHERE id=?", (run_id,))


def active_runs() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM runs WHERE status IN ('running', 'stopping', 'paused')"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------- models --
_MODEL_COLS = (
    "id", "name", "run_id", "agent", "env", "path", "size_bytes",
    "best_eval_reward", "final_mean_reward", "metrics_json", "description",
    "created_at", "tags_json", "checkpoint_type", "checksum",
    "version", "is_favorite", "is_archived", "architecture_json"
)


def keep_model(
    run_id: str,
    source_path: Path,
    name: Optional[str] = None,
    description: str = "",
    checkpoint_type: str = "best",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    import shutil

    run = get_run(run_id)
    if not run:
        raise KeyError(f"run {run_id} not found")
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(source_path)

    mid = new_model_id()
    model_name = name or f"{run['agent']}-{run['env']}-{mid}"
    dest = MODELS_DIR / f"{mid}.pth"
    shutil.copy2(src, dest)

    checksum = compute_sha256(dest)

    architecture = {}
    if "hidden_size" in run.get("params", {}):
        architecture["hidden_size"] = run["params"]["hidden_size"]

    record = {
        "id": mid,
        "name": model_name,
        "run_id": run_id,
        "agent": run["agent"],
        "env": run["env"],
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "best_eval_reward": run.get("best_eval_reward"),
        "final_mean_reward": run.get("final_mean_reward"),
        "metrics_json": json.dumps(
            {
                "seed": run["seed"],
                "episodes_done": run.get("episodes_done"),
                "device": run.get("device", "cpu"),
            }
        ),
        "description": description,
        "created_at": now(),
        "tags_json": json.dumps(tags or []),
        "checkpoint_type": checkpoint_type,
        "checksum": checksum,
        "version": "1.0",
        "is_favorite": 0,
        "is_archived": 0,
        "architecture_json": json.dumps(architecture),
    }

    with _conn() as con:
        con.execute(
            """
            INSERT INTO models (
                id, name, run_id, agent, env, path, size_bytes, best_eval_reward,
                final_mean_reward, metrics_json, description, created_at,
                tags_json, checkpoint_type, checksum, version, is_favorite,
                is_archived, architecture_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            tuple(record[c] for c in _MODEL_COLS),
        )
        con.execute("UPDATE runs SET model_kept=1 WHERE id=?", (run_id,))

    record["metrics"] = json.loads(record.pop("metrics_json"))
    record["tags"] = json.loads(record.pop("tags_json"))
    record["architecture"] = json.loads(record.pop("architecture_json"))
    return record


def import_model_file(
    file_bytes: bytes,
    agent: str,
    env: str,
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    best_eval_reward: Optional[float] = None,
) -> Dict[str, Any]:
    mid = new_model_id()
    model_name = name or f"imported-{agent}-{env}-{mid}"
    dest = MODELS_DIR / f"{mid}.pth"
    dest.write_bytes(file_bytes)

    checksum = compute_sha256(dest)

    record = {
        "id": mid,
        "name": model_name,
        "run_id": None,
        "agent": agent,
        "env": env,
        "path": str(dest),
        "size_bytes": len(file_bytes),
        "best_eval_reward": best_eval_reward,
        "final_mean_reward": None,
        "metrics_json": json.dumps({"imported": True}),
        "description": description,
        "created_at": now(),
        "tags_json": json.dumps(tags or ["imported"]),
        "checkpoint_type": "imported",
        "checksum": checksum,
        "version": "1.0",
        "is_favorite": 0,
        "is_archived": 0,
        "architecture_json": json.dumps({}),
    }

    with _conn() as con:
        con.execute(
            """
            INSERT INTO models (
                id, name, run_id, agent, env, path, size_bytes, best_eval_reward,
                final_mean_reward, metrics_json, description, created_at,
                tags_json, checkpoint_type, checksum, version, is_favorite,
                is_archived, architecture_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            tuple(record[c] for c in _MODEL_COLS),
        )

    record["metrics"] = json.loads(record.pop("metrics_json"))
    record["tags"] = json.loads(record.pop("tags_json"))
    record["architecture"] = json.loads(record.pop("architecture_json"))
    return record


def update_model(model_id: str, **fields):
    if not fields:
        return
    cols, vals = [], []
    for k, v in fields.items():
        if k in ("tags", "metrics", "architecture"):
            k_json = f"{k}_json"
            cols.append(f"{k_json}=?")
            vals.append(json.dumps(v))
            continue
        if k not in _MODEL_COLS:
            raise ValueError(f"unknown model column {k}")
        if k in ("tags_json", "metrics_json", "architecture_json") and isinstance(v, (dict, list)):
            v = json.dumps(v)
        cols.append(f"{k}=?")
        vals.append(v)
    vals.append(model_id)
    with _conn() as con:
        con.execute(f"UPDATE models SET {', '.join(cols)} WHERE id=?", vals)


def list_models(include_archived: bool = True) -> List[Dict[str, Any]]:
    query = "SELECT * FROM models"
    if not include_archived:
        query += " WHERE is_archived=0"
    query += " ORDER BY is_favorite DESC, created_at DESC"

    with _conn() as con:
        rows = con.execute(query).fetchall()
    out = []
    for r in rows:
        d = _row_to_dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json", "{}") or "{}")
        d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
        d["architecture"] = json.loads(d.pop("architecture_json", "{}") or "{}")
        d["is_favorite"] = bool(d.get("is_favorite", 0))
        d["is_archived"] = bool(d.get("is_archived", 0))
        out.append(d)
    return out


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    if not row:
        return None
    d = _row_to_dict(row)
    d["metrics"] = json.loads(d.pop("metrics_json", "{}") or "{}")
    d["tags"] = json.loads(d.pop("tags_json", "[]") or "[]")
    d["architecture"] = json.loads(d.pop("architecture_json", "{}") or "{}")
    d["is_favorite"] = bool(d.get("is_favorite", 0))
    d["is_archived"] = bool(d.get("is_archived", 0))
    return d


def delete_model(model_id: str):
    m = get_model(model_id)
    if not m:
        return
    try:
        Path(m["path"]).unlink(missing_ok=True)
    except OSError:
        pass
    with _conn() as con:
        con.execute("DELETE FROM models WHERE id=?", (model_id,))
        if m.get("run_id"):
            con.execute(
                "UPDATE runs SET model_kept=0 WHERE id=? AND model_kept=1 AND NOT EXISTS"
                " (SELECT 1 FROM models WHERE run_id=?)",
                (m["run_id"], m["run_id"]),
            )


# ------------------------------------------------------------------ misc --
def reconcile_on_startup(active_pids: set[int]):
    """Mark runs whose worker process is gone as 'interrupted'."""
    for run in active_runs():
        pid = run.get("pid")
        alive = False
        if pid and pid in active_pids:
            alive = True
        elif pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if not alive:
            update_run(run["id"], status="interrupted", finished_at=now())


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("params_json", "config_json", "tags_json"):
        if key in d:
            clean_key = key.replace("_json", "")
            d[clean_key] = json.loads(d.pop(key) or ("[]" if "tags" in key else "{}"))
    for key in ("created_at", "started_at", "finished_at"):
        if d.get(key):
            d[f"{key}_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d[key]))
    return d
