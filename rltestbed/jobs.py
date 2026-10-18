"""Job manager: launches training workers as subprocesses and controls them."""

import os
import signal
import subprocess
import sys
from typing import Dict, Optional

from rltestbed.registry import get_run, update_run, active_runs, reconcile_on_startup


def _default_limit() -> int:
    env = os.environ.get("RLTESTBED_MAX_CONCURRENT")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(2, (os.cpu_count() or 2) // 2)


class JobManager:
    """Starts training workers, honouring a concurrency limit.

    Runs beyond the limit wait with status ``queued`` and are started by
    :meth:`pump`, which the server calls about once a second. Set
    ``RLTESTBED_MAX_CONCURRENT`` to change the limit.
    """

    def __init__(self):
        self._procs: Dict[str, subprocess.Popen] = {}
        self.max_concurrent = _default_limit()

    # ------------------------------------------------------------------ #
    def running_count(self) -> int:
        return sum(1 for rid in list(self._procs) if self.is_running(rid))

    def enqueue(self, run_id: str) -> str:
        """Start now if there is capacity, otherwise mark ``queued``."""
        if self.running_count() < self.max_concurrent:
            self.start(run_id)
            return "running"
        update_run(run_id, status="queued")
        return "queued"

    def pump(self) -> int:
        """Start queued runs while capacity allows. Returns how many started."""
        from rltestbed.registry import list_runs

        started = 0
        free = self.max_concurrent - self.running_count()
        if free <= 0:
            return 0
        queued = sorted(list_runs(status="queued", limit=500), key=lambda r: r["created_at"])
        for run in queued[:free]:
            self.start(run["id"])
            started += 1
        return started

    # ------------------------------------------------------------------ #
    def start(self, run_id: str) -> bool:
        run = get_run(run_id)
        if not run:
            raise KeyError(run_id)
        if run_id in self._procs and self._procs[run_id].poll() is None:
            return False  # already running

        cmd = [
            sys.executable,
            "-m",
            "rltestbed.cli",
            "train-worker",
            "--run-id",
            run_id,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            cwd=str(_project_root()),
            start_new_session=True,  # own process group -> clean interrupt
        )
        self._procs[run_id] = proc
        update_run(run_id, pid=proc.pid, status="running")
        return True

    def stop(self, run_id: str, timeout: float = 10.0) -> bool:
        """Graceful interrupt: SIGTERM lets the trainer finish the current
        episode, write checkpoints and mark the run 'stopped'."""
        proc = self._procs.get(run_id)
        if proc is None or proc.poll() is not None:
            # maybe a CLI-started process; try by recorded pid
            run = get_run(run_id)
            pid = run.get("pid") if run else None
            if not pid:
                return False
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except OSError:
                return False
            update_run(run_id, status="stopping")
            return True

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()

        update_run(run_id, status="stopping")
        try:
            proc.wait(timeout=timeout)
            # worker exited cleanly and wrote its own terminal status
            self._sync_finished(run_id)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            update_run(run_id, status="stopped", finished_at=_now(), pid=None)
        return True

    def is_running(self, run_id: str) -> bool:
        proc = self._procs.get(run_id)
        if proc is not None:
            if proc.poll() is None:
                return True
            del self._procs[run_id]
            self._sync_finished(run_id)
            return False
        # external process? check DB status + liveness
        for run in active_runs():
            if run["id"] == run_id:
                return True
        return False

    def _sync_finished(self, run_id: str):
        """Worker exited on its own; mirror the trainer's progress status into the DB."""
        import json as _json
        from rltestbed.config import run_dir as run_dir_for

        progress_path = run_dir_for(run_id) / "progress.json"
        status = "failed"
        if progress_path.exists():
            try:
                data = _json.loads(progress_path.read_text())
                status = data.get("status", "completed")
                if status in ("running", "stopping"):
                    status = "stopped" if status == "stopping" else "completed"
            except Exception:
                pass
        update_run(run_id, status=status, finished_at=_now(), pid=None)

    def reconcile(self):
        reconcile_on_startup({p.pid for p in self._procs.values() if p.poll() is None})


def _now():
    import time

    return time.time()


def _project_root():
    from rltestbed.config import PROJECT_ROOT

    return PROJECT_ROOT


JOB_MANAGER = JobManager()
