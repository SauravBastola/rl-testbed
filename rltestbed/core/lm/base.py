"""Pretraining: the base model every RL run starts from.

Large-language-model RL never starts from scratch: a pretrained (and usually
supervised fine-tuned) model already produces well-formed, partly correct
answers, and RL sharpens it. Here each task's base model is trained on
demonstrations corrupted with probability ``noise``, so it learns the format
but is only partly right. It is cached under the models directory, keyed by
task and settings, and doubles as the frozen reference policy for KL penalties.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from rltestbed.core.lm.model import EOS, TinyGPT, completion_mask, encode
from rltestbed.core.lm.tasks import TASKS, Task

BASE_SETTINGS = {"batch": 128, "lr": 3e-3, "noise": 0.5, "seed": 1234}
# Train to convergence on demonstrations that are wrong half the time: the
# model then knows the answer (greedy decoding is nearly always right) but a
# single sample is right only about half the time. Sharpening that is what RL
# with verifiable rewards does. Stopping mid-way through learning instead gave
# a base accuracy that varied wildly between machines.
BASE_STEPS = {"lm_addition": 600, "lm_addition_hackable": 600, "lm_reverse": 200, "lm_sort": 200}
VERSION = 2


def batch_tensors(task: Task, pairs: List[Tuple[str, str]]) -> torch.Tensor:
    """Prompt + completion + EOS, padded with EOS to the task's full length."""
    rows = []
    for prompt, completion in pairs:
        ids = encode(prompt) + encode(completion) + [EOS]
        ids += [EOS] * (task.prompt_len + task.max_new - len(ids))
        rows.append(ids[: task.prompt_len + task.max_new])
    return torch.tensor(rows, dtype=torch.long)


def sft_loss(model: TinyGPT, seqs: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Cross-entropy on completion tokens only (through the first EOS)."""
    logits = model(seqs[:, :-1])[:, prompt_len - 1:, :]
    targets = seqs[:, prompt_len:]
    mask = completion_mask(seqs, prompt_len)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           reduction="none").view_as(targets)
    return (loss * mask).sum() / mask.sum()


def train_sft(model: TinyGPT, task: Task, steps: int, batch: int, lr: float, noise: float,
              rng: random.Random, log=None) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss = torch.tensor(0.0)
    for step in range(steps):
        pairs = []
        for _ in range(batch):
            prompt, answer = task.sample(rng)
            pairs.append((prompt, task.demo(answer, rng, noise)))
        loss = sft_loss(model, batch_tensors(task, pairs), task.prompt_len)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and (step + 1) % 100 == 0:
            log(f"pretraining step {step + 1}/{steps}: loss {loss.item():.3f}")
    return float(loss.item())


def _cache_path(task_name: str, settings: Dict) -> Path:
    from rltestbed.config import MODELS_DIR

    key = "-".join(f"{k}{settings[k]}" for k in sorted(settings))
    return Path(MODELS_DIR) / "lm_base" / f"{task_name}-v{VERSION}-{key}.pt"


def base_model(task_name: str, settings: Dict = None, log=None) -> TinyGPT:
    """The task's pretrained base model, trained once and cached."""
    settings = {**BASE_SETTINGS, "steps": BASE_STEPS.get(task_name, 200), **(settings or {})}
    path = _cache_path(task_name, settings)
    model = TinyGPT()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
    if log:
        log(f"Pretraining the {task_name} base model once ({settings['steps']} steps); "
            "later runs reuse it.")
    torch.manual_seed(settings["seed"])
    train_sft(model, TASKS[task_name], settings["steps"], settings["batch"], settings["lr"],
              settings["noise"], random.Random(settings["seed"]), log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)  # atomic: concurrent runs never read a half-written file
    return model
