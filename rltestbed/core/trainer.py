"""Training loop with TensorBoard logging, periodic evaluation, checkpointing,
live progress reporting, structured log capture, early stopping and graceful interruption.

Runs inside a subprocess launched by the job manager; communicates state via:
    runs/<run_id>/progress.json
    runs/<run_id>/history.jsonl
    runs/<run_id>/worker.log
"""

import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class GracefulStop(Exception):
    pass


def _install_signal_handlers():
    import threading

    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, frame):
        raise GracefulStop(f"signal {signum}")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


class Trainer:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        env_name: str,
        agent_name: str,
        params: Dict,
        seed: int = 42,
        episodes: int = 500,
        max_steps: Optional[int] = None,
        eval_freq: int = 25,
        eval_episodes: int = 10,
        checkpoint_freq: int = 100,
        device: str = "cpu",
        save_best: bool = True,
        save_final: bool = True,
        auto_eval: bool = True,
        early_stopping_patience: Optional[int] = None,
        log_wandb: bool = False,
    ):
        from rltestbed.core.agents import create_agent
        from rltestbed.core.envs import get_env_spec, make_env

        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.env_name = env_name
        self.agent_name = agent_name
        self.params = dict(params)
        self.seed = seed
        self.episodes = episodes
        spec = get_env_spec(env_name)
        self.max_steps = max_steps or spec.max_steps
        self.eval_freq = max(eval_freq, 1)
        self.eval_episodes = eval_episodes
        self.checkpoint_freq = max(checkpoint_freq, 1)
        self.device = device
        self.save_best = save_best
        self.save_final = save_final
        self.auto_eval = auto_eval
        self.early_stopping_patience = early_stopping_patience
        self.log_wandb = log_wandb

        # Logger setup: file + stdout
        self.log_path = self.run_dir / "worker.log"
        self._setup_logger()

        # Reproducibility
        import random
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        self.logger.info(f"Initialising run {run_id}: {agent_name} on {env_name} (seed={seed}, device={device})")

        self.env = make_env(env_name)
        self.agent = create_agent(
            agent_name, self.env.obs_dim, self.env.act_dim, self.params, device=self.device
        )

        # Continuous-control agents emit actions in a normalised range and
        # rescale using the environment's action limits.
        low = getattr(self.env, "act_low", None)
        high = getattr(self.env, "act_high", None)
        if low is not None and high is not None:
            try:
                self.agent.set_action_bounds(low, high)
            except Exception:  # pragma: no cover - agent ignores bounds
                self.logger.debug("agent does not support action bounds")

        # Artifacts
        self.tb_dir = self.run_dir / "tb"
        self.progress_path = self.run_dir / "progress.json"
        self.history_path = self.run_dir / "history.jsonl"
        self.checkpoint_path = self.run_dir / "checkpoint.pth"
        self.best_path = self.run_dir / "best.pth"

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        except Exception as e:
            self.logger.warning(f"TensorBoard writer not initialised: {e}")
            self.writer = None

        # Weights & Biases (optional)
        self._wandb_run = None
        if self.log_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project="rl-testbed",
                    name=f"{agent_name}-{env_name}-{run_id}",
                    config={"seed": seed, "episodes": episodes, **params},
                    reinit=True,
                )
            except Exception as e:
                self.logger.warning(f"W&B logging requested but failed to init: {e}")

        self.status = "running"
        self.episode_rewards = []
        self.eval_history = []
        self.best_eval_reward = -float("inf")
        self._no_improvement_count = 0
        self._stop_requested = False
        self._start_time = time.time()
        self.total_env_steps = 0

    def _setup_logger(self):
        self.logger = logging.getLogger(f"trainer_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # reset

        # File handler
        fh = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

        # Stream handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(sh)

    # ------------------------------------------------------------------ #
    def _write_progress(self, episode, metrics, extra=None):
        elapsed = time.time() - self._start_time
        eps_rate = episode / max(elapsed, 1e-6)
        fps = self.total_env_steps / max(elapsed, 1e-6)

        data = {
            "run_id": self.run_id,
            "status": self.status,
            "episode": episode,
            "total_episodes": self.episodes,
            "total_env_steps": self.total_env_steps,
            "episode_rewards_tail": self.episode_rewards[-200:],
            "running_reward": (
                float(np.mean(self.episode_rewards[-100:]))
                if self.episode_rewards
                else None
            ),
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else None,
            "best_eval_reward": (
                self.best_eval_reward if np.isfinite(self.best_eval_reward) else None
            ),
            "eval_history": self.eval_history[-50:],
            "agent_metrics": metrics,
            "elapsed_sec": round(elapsed, 1),
            "episodes_per_sec": round(eps_rate, 2),
            "fps": round(fps, 1),
            "device": self.device,
            "updated_at": time.time(),
        }
        if extra:
            data.update(extra)
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.progress_path)

    def _append_history(self, record):
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def stop(self):
        self._stop_requested = True

    # ------------------------------------------------------------------ #
    def _reset(self):
        obs = self.env.reset()
        if not isinstance(obs, dict):
            obs = {
                "obs": list(np.atleast_1d(np.asarray(obs, dtype=np.float32))),
                "action_mask": [1] * self.env.act_dim,
            }
        return obs

    def _evaluate_greedy(self):
        rewards = []
        was_training = self.agent.TRAIN
        self.agent.TRAIN = False
        for _ in range(self.eval_episodes):
            obs = self._reset()
            total, steps, done = 0.0, 0, False
            while not done and steps < self.max_steps:
                action = self.agent.get_action(obs)
                obs, r, terminated, truncated, info = self.env.step(action)
                total += float(r)
                steps += 1
                done = terminated or truncated
            rewards.append(total)
        self.agent.TRAIN = was_training
        return {
            "episode_mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        }

    # ------------------------------------------------------------------ #
    def run(self) -> Dict:
        _install_signal_handlers()
        self._start_time = time.time()
        self.status = "running"

        try:
            self._loop()
            if self.status == "stopping":
                self.status = "stopped"
            elif self.status == "running":
                self.status = "completed"
        except GracefulStop as gs:
            self.logger.info(f"Training interrupted cleanly ({gs})")
            self.status = "stopped"
        except Exception as e:
            import traceback

            self.logger.error(f"Training failed with exception: {e}\n{traceback.format_exc()}")
            self.status = "failed"
            self._write_progress(
                episode=len(self.episode_rewards),
                metrics={},
                extra={"error": f"{type(e).__name__}: {e}"},
            )
            raise
        finally:
            self._finalize()

        return self.summary()

    def _loop(self):
        for episode in range(1, self.episodes + 1):
            if self._stop_requested:
                self.status = "stopping"
                break

            # Progress hook for LR / annealing
            self.agent.set_training_progress(episode / self.episodes)

            obs = self._reset()
            ep_reward, steps, done = 0.0, 0, False
            while not done and steps < self.max_steps:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                # World models must not learn that a time limit ends the world:
                # agents that care can read whether this step was only a truncation.
                self.agent.last_step_truncated = bool(truncated and not terminated)
                self.agent.update(
                    obs, action, reward, next_obs, terminated or truncated
                )
                obs = next_obs
                ep_reward += float(reward)
                steps += 1
                self.total_env_steps += 1
                done = terminated or truncated

            self.agent.end_of_episode()
            self.episode_rewards.append(ep_reward)
            # Agents may publish JSON artifacts for the run page (e.g. a world
            # model's imagined vs real trajectory).
            publish = getattr(self.agent, "artifacts", None)
            if callable(publish):
                try:
                    for fname, data in (publish() or {}).items():
                        if isinstance(data, dict) and "real" in data and "labels" not in data:
                            from rltestbed.core.envs.observations import labels_for

                            data = {**data, "labels": labels_for(
                                self.env_name, self.env, self.env.obs_dim)}
                        (self.run_dir / fname).write_text(json.dumps(data))
                except Exception as e:  # never let an artifact stop training
                    self.logger.warning(f"Could not write agent artifact: {e}")

            metrics = self.agent.get_metrics()

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("train/episode_reward", ep_reward, episode)
                self.writer.add_scalar(
                    "train/mean_reward_100",
                    float(np.mean(self.episode_rewards[-100:])),
                    episode,
                )
                self.writer.add_scalar("train/episode_length", steps, episode)
                for k, v in metrics.items():
                    try:
                        self.writer.add_scalar(f"agent/{k}", v, episode)
                    except Exception:
                        pass

            # W&B logging
            if self._wandb_run:
                try:
                    import wandb

                    wandb.log(
                        {
                            "episode": episode,
                            "train/episode_reward": ep_reward,
                            "train/mean_reward_100": float(np.mean(self.episode_rewards[-100:])),
                            **{f"agent/{k}": v for k, v in metrics.items()},
                        }
                    )
                except Exception:
                    pass

            record = {
                "episode": episode,
                "reward": ep_reward,
                "steps": steps,
                "mean_reward": float(np.mean(self.episode_rewards[-100:])),
                **{f"metric_{k}": v for k, v in metrics.items()},
            }

            # Periodic evaluation + best checkpoint
            if self.auto_eval and (episode % self.eval_freq == 0 or episode == self.episodes):
                eval_res = self._evaluate_greedy()
                eval_res["episode"] = episode
                self.eval_history.append(eval_res)
                record["eval"] = eval_res
                mean_eval = eval_res["episode_mean"]

                self.logger.info(
                    f"Ep {episode:5d}/{self.episodes} | Train R={ep_reward:7.2f} (mean100={record['mean_reward']:7.2f}) | "
                    f"Eval={mean_eval:7.2f} ± {eval_res['std']:5.2f}"
                )

                if self.writer:
                    self.writer.add_scalar("eval/mean_reward", mean_eval, episode)
                    self.writer.add_scalar("eval/std", eval_res["std"], episode)

                if self._wandb_run:
                    try:
                        import wandb

                        wandb.log({"episode": episode, "eval/mean_reward": mean_eval})
                    except Exception:
                        pass

                if mean_eval > self.best_eval_reward:
                    self.best_eval_reward = mean_eval
                    self._no_improvement_count = 0
                    if self.save_best:
                        self.agent.save_model(str(self.best_path))
                        self.logger.info(f"-> Saved new best model checkpoint (eval={mean_eval:.2f})")
                else:
                    self._no_improvement_count += 1

                # Early stopping
                if (
                    self.early_stopping_patience is not None
                    and self._no_improvement_count >= self.early_stopping_patience
                ):
                    self.logger.info(
                        f"Early stopping triggered: no eval improvement for {self._no_improvement_count} checks."
                    )
                    break

            # Periodic checkpointing
            if episode % self.checkpoint_freq == 0 and self.save_final:
                try:
                    self.agent.save_model(str(self.checkpoint_path))
                except Exception as e:
                    self.logger.warning(f"Failed to write periodic checkpoint: {e}")

            self._append_history(record)

            if (
                episode % 5 == 0
                or episode == 1
                or episode == self.episodes
                or self._stop_requested
            ):
                self._write_progress(episode=episode, metrics=metrics)

    def _finalize(self):
        try:
            self.agent.finalize_training()
        except Exception as e:
            self.logger.error(f"Failed to flush pending training data: {e}")

        if self.save_final and self.episode_rewards:
            try:
                self.agent.save_model(str(self.checkpoint_path))
                self.logger.info("Saved final checkpoint.")
            except Exception as e:
                self.logger.error(f"Failed to save final checkpoint: {e}")

        self._write_progress(
            episode=len(self.episode_rewards),
            metrics=self.agent.get_metrics(),
        )
        if self.writer:
            self.writer.close()
        if self._wandb_run:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        self.env.close()
        self.logger.info(f"Run {self.run_id} finalised with status: {self.status}")

    def summary(self) -> Dict:
        last100 = self.episode_rewards[-100:]
        return {
            "episodes_done": len(self.episode_rewards),
            "final_mean_reward": float(np.mean(last100)) if last100 else 0.0,
            "best_eval_reward": (
                float(self.best_eval_reward)
                if np.isfinite(self.best_eval_reward)
                else None
            ),
            "eval_history": self.eval_history,
            "status": self.status,
            "total_env_steps": self.total_env_steps,
        }
