"""Training loop for language-model RL, compatible with the rest of the testbed.

It writes the same files as :class:`rltestbed.core.trainer.Trainer`
(``progress.json``, ``history.jsonl``, ``worker.log``, checkpoints) and
honours the same stop signals, so live charts, comparisons, sweeps, experiment
files and the learning path work unchanged. One "episode" in the history is
one training step (a batch of prompts and their completions).

Evaluation samples one answer per held-out prompt at temperature 1 and reports
the mean reward: what RL with verifiable rewards improves is how often a single
sample is right. Greedy accuracy is logged alongside as a metric.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from rltestbed.core.lm.base import base_model
from rltestbed.core.lm.model import decode, encode, sample
from rltestbed.core.lm.tasks import TASKS
from rltestbed.core.trainer import GracefulStop, _install_signal_handlers

EVAL_PROMPTS = 200
SAMPLE_PREVIEW = 12


class SequenceTrainer:
    def __init__(self, run_id: str, run_dir: Path, env_name: str, agent_name: str, params: Dict,
                 seed: int = 42, episodes: int = 200, eval_freq: int = 10, eval_episodes: int = 10,
                 save_best: bool = True, save_final: bool = True,
                 early_stopping_patience: Optional[int] = None, **_ignored):
        from rltestbed.core.agents import create_agent

        self.run_id, self.run_dir = run_id, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.env_name, self.agent_name, self.params = env_name, agent_name, dict(params)
        self.seed, self.episodes = seed, episodes
        self.eval_freq = max(1, eval_freq)
        self.save_best, self.save_final = save_best, save_final
        self.patience = early_stopping_patience
        self.task = TASKS[env_name]
        self.progress_path = self.run_dir / "progress.json"
        self.history_path = self.run_dir / "history.jsonl"
        self.best_path = self.run_dir / "best.pth"
        self.checkpoint_path = self.run_dir / "checkpoint.pth"

        self.logger = logging.getLogger(f"rltestbed.lm.{run_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(self.run_dir / "worker.log")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                              "%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(fh)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.rng = random.Random(seed)
        self.agent = create_agent(agent_name, 1, 1, self.params)
        self.agent.attach(base_model(env_name, log=self.logger.info))
        # held-out prompts, fixed across runs so evaluations are comparable
        eval_rng = random.Random(10_000)
        self.eval_set = [self.task.sample(eval_rng) for _ in range(EVAL_PROMPTS)]

        self.rewards, self.eval_history = [], []
        self.best_eval = -float("inf")
        self.status, self.total_samples = "created", 0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        model = self.agent.policy
        model.eval()
        prompts = torch.tensor([encode(p) for p, _ in self.eval_set])
        gen = torch.Generator().manual_seed(7)
        torch.manual_seed(gen.initial_seed())
        sampled = sample(model, prompts, self.task.max_new)
        greedy = sample(model, prompts, self.task.max_new, greedy=True)
        model.train()
        rows, rewards, correct, greedy_ok = [], [], [], []
        for (prompt, answer), s, g in zip(self.eval_set, sampled, greedy):
            text = decode(s[self.task.prompt_len:])
            r, extra = self.task.reward(text, answer)
            rewards.append(r)
            correct.append(extra["accuracy"])
            greedy_ok.append(float(self.task.correct(decode(g[self.task.prompt_len:]), answer)))
            if len(rows) < SAMPLE_PREVIEW:
                rows.append({"prompt": prompt, "completion": text, "answer": answer,
                             "reward": r, "correct": bool(extra["accuracy"])})
        (self.run_dir / "samples.json").write_text(json.dumps(rows))
        return {"episode_mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
                "min": float(np.min(rewards)), "max": float(np.max(rewards)),
                "accuracy": float(np.mean(correct)), "greedy_accuracy": float(np.mean(greedy_ok))}

    def _progress(self, step, metrics):
        elapsed = time.time() - self._start
        data = {
            "run_id": self.run_id, "status": self.status, "episode": step,
            "total_episodes": self.episodes, "total_env_steps": self.total_samples,
            "episode_rewards_tail": self.rewards[-200:],
            "running_reward": float(np.mean(self.rewards[-20:])) if self.rewards else None,
            "last_reward": self.rewards[-1] if self.rewards else None,
            "best_eval_reward": self.best_eval if np.isfinite(self.best_eval) else None,
            "eval_history": self.eval_history[-50:], "agent_metrics": metrics,
            "elapsed_sec": round(elapsed, 1), "episodes_per_sec": round(step / max(elapsed, 1e-6), 2),
            "fps": round(self.total_samples / max(elapsed, 1e-6), 1), "device": "cpu",
            "updated_at": time.time(),
        }
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.progress_path)

    def run(self) -> Dict:
        _install_signal_handlers()
        self._start = time.time()
        self.status = "running"
        self.logger.info(f"{self.agent_name} on {self.env_name}: {self.episodes} steps")
        base = self.evaluate()
        self.logger.info(f"Base model: sampled reward {base['episode_mean']:.3f}, "
                         f"accuracy {base['accuracy']:.3f}, greedy {base['greedy_accuracy']:.3f}")
        metrics: Dict[str, float] = {}
        stale = 0
        try:
            for step in range(1, self.episodes + 1):
                metrics = self.agent.train_step(self.task, self.rng)
                self.total_samples += self.params.get("prompts_per_step", 1) * self.params.get("group_size", 1)
                self.rewards.append(float(metrics["reward"]))
                record = {"episode": step, "reward": metrics["reward"],
                          "mean_reward": float(np.mean(self.rewards[-20:])),
                          "steps": metrics.get("length", self.task.answer_len)}
                record.update({f"metric_{k}": float(v) for k, v in metrics.items() if k != "reward"})
                if step % self.eval_freq == 0 or step == self.episodes:
                    ev = self.evaluate()
                    record["eval"] = ev
                    record["metric_eval_accuracy"] = ev["accuracy"]
                    record["metric_greedy_accuracy"] = ev["greedy_accuracy"]
                    self.eval_history.append({"episode": step, **ev})
                    self.logger.info(f"Step {step:5d}/{self.episodes} | train reward "
                                     f"{metrics['reward']:.3f} | eval {ev['episode_mean']:.3f} "
                                     f"(accuracy {ev['accuracy']:.3f})")
                    if ev["episode_mean"] > self.best_eval:
                        self.best_eval, stale = ev["episode_mean"], 0
                        if self.save_best:
                            self.agent.save_model(self.best_path)
                    else:
                        stale += 1
                        if self.patience and stale >= self.patience:
                            self.logger.info("Early stopping: evaluation plateaued.")
                            break
                with open(self.history_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                if step % 5 == 0 or step == 1:
                    self._progress(step, metrics)
            self.status = "completed"
        except (GracefulStop, KeyboardInterrupt):
            self.status = "stopped"
            self.logger.info("Stopped: finishing up.")
        except Exception:
            self.status = "failed"
            self.logger.exception("Training failed")
            raise
        finally:
            if self.save_final and self.rewards:
                self.agent.save_model(self.checkpoint_path)
            self._progress(len(self.rewards), metrics)
        return self.summary()

    def summary(self) -> Dict:
        return {"episodes_done": len(self.rewards),
                "final_mean_reward": float(np.mean(self.rewards[-20:])) if self.rewards else 0.0,
                "best_eval_reward": self.best_eval if np.isfinite(self.best_eval) else None,
                "eval_history": self.eval_history, "status": self.status,
                "total_env_steps": self.total_samples}
