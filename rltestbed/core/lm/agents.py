"""RL fine-tuning algorithms for language models.

All four share one loop shape, driven by :class:`~rltestbed.core.lm.trainer.SequenceTrainer`:
sample prompts, generate completions with the current policy, score them with
the task's verifier, update. They differ in how the score becomes a gradient:

- SFT          imitate correct answers (no reward: the supervised baseline)
- REINFORCE    sequence log-likelihood x (reward - batch mean), plus KL
- GRPO         G samples per prompt, advantages relative to the group, PPO-style
               clipped ratios over several epochs, plus KL to a frozen reference
- DPO          no reward in the loss at all: prefer the better of two samples by
               a margin measured against the reference model

The KL term uses the k3 estimator  exp(r) - r - 1,  r = log π_ref - log π_θ,
which is unbiased and always non-negative (Schulman, 2020).
"""

import random
from typing import Any, Dict, List

import torch

from rltestbed.core.agents.base import Agent, ParamSpec, build_spec
from rltestbed.core.lm.base import batch_tensors, sft_loss
from rltestbed.core.lm.model import (TinyGPT, completion_logprobs, completion_mask, decode,
                                     encode, masked_mean, sample)
from rltestbed.core.lm.tasks import Task

TOKEN_SUPPORT = dict(obs_types=["tokens"], action_types=["token"])


def _common_params(lr=1e-3, batch=32):
    return [
        ParamSpec("lr", "Learning rate", lr, "float", "Adam learning rate for the policy.",
                  min=1e-6, max=1e-1),
        ParamSpec("prompts_per_step", "Prompts per step", batch, "int",
                  "Prompts sampled each training step.", min=1, max=512),
        ParamSpec("temperature", "Sampling temperature", 1.0, "float",
                  "Temperature for generating training completions.", min=0.1, max=2.0,
                  advanced=True),
    ]


KL_PARAM = ParamSpec("kl_coef", "KL coefficient β", 0.04, "float",
                     "Penalty for drifting from the frozen base (reference) model. 0 removes "
                     "it, which invites reward hacking.", min=0.0, max=10.0)


class LMAgent(Agent):
    """Base for language-model agents: holds the policy, reference and optimiser."""

    def __init__(self, obs_dim: int, act_dim: int, params: Dict[str, Any], device: str = "cpu"):
        super().__init__(act_dim, params, device)
        self.policy: TinyGPT = None
        self.ref: TinyGPT = None
        self.optim = None

    def attach(self, base: TinyGPT):
        """Start from the pretrained base; keep a frozen copy as the reference."""
        self.policy = base
        self.ref = TinyGPT()
        self.ref.load_state_dict(base.state_dict())
        self.ref.requires_grad_(False)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.params["lr"])

    # Agent interface (not used: language agents are driven by SequenceTrainer)
    def get_action(self, obs, valid_actions=None):
        raise NotImplementedError("language agents are trained with SequenceTrainer")

    def update(self, obs, action, reward, next_obs, done):
        raise NotImplementedError("language agents are trained with SequenceTrainer")

    def get_metrics(self):
        return {}  # per-step metrics come back from train_step

    def save_model(self, path):
        torch.save({"format": "tiny-gpt-v1", "agent_name": self.SPEC["name"],
                    "policy": self.policy.state_dict(), "params": self.params}, str(path))

    def load_model(self, path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if self.policy is None:
            self.attach(TinyGPT())
        self.policy.load_state_dict(ckpt["policy"])

    # ------------------------------------------------------------------ #
    def rollouts(self, task: Task, rng: random.Random, n_prompts: int, group: int = 1):
        """Sample prompts, generate ``group`` completions each, score them."""
        pairs = [task.sample(rng) for _ in range(n_prompts)]
        prompts = torch.tensor([encode(p) for p, _ in pairs for _ in range(group)])
        self.policy.eval()
        seqs = sample(self.policy, prompts, task.max_new, self.params.get("temperature", 1.0))
        self.policy.train()
        texts = [decode(s[task.prompt_len:]) for s in seqs]
        answers = [a for _, a in pairs for _ in range(group)]
        scored = [task.reward(t, a) for t, a in zip(texts, answers)]
        rewards = torch.tensor([r for r, _ in scored], dtype=torch.float32)
        extras = {k: sum(e[k] for _, e in scored) / len(scored) for k in scored[0][1]}
        return seqs, rewards, extras, texts

    def kl_to_ref(self, seqs, prompt_len, logp, mask):
        with torch.no_grad():
            ref_logp, _ = completion_logprobs(self.ref, seqs, prompt_len)
        r = ref_logp - logp
        return masked_mean(torch.exp(r) - r - 1.0, mask)

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()

    def train_step(self, task: Task, rng: random.Random) -> Dict[str, float]:
        raise NotImplementedError


# ---------------------------------------------------------------------- SFT
class SFTAgent(LMAgent):
    SPEC = build_spec(
        name="lm_sft", label="Supervised fine-tuning", category="language",
        policy_type="offline", **TOKEN_SUPPORT,
        description="Fine-tune on correct answers by maximum likelihood. Not RL: the "
        "baseline every RL method for language models is compared against.",
        params=_common_params(lr=1e-3),
        beginner="Show the model the right answer and nudge it to say exactly that. "
        "Needs a correct answer for every prompt, which RL does not.",
        technical="Minimise the cross-entropy of reference answers (demonstrations with "
        "no noise) given the prompt. The reward is never used.",
        intuition="Learning by copying a teacher.",
        update_rule="L = −Σ_t log π_θ(y_t | x, y_<t)",
        learns="The policy's next-token distribution.",
        failure_modes=["Needs labelled answers", "Learns only as well as its demonstrations"],
        expected_metrics=[{"key": "loss", "meaning": "Cross-entropy on correct answers."},
                          {"key": "accuracy", "meaning": "Fraction of sampled answers that are correct."}],
        recommended_envs=["lm_addition", "lm_sort"], related=["grpo", "dpo"],
    )

    def train_step(self, task, rng):
        pairs = [task.sample(rng) for _ in range(self.params["prompts_per_step"])]
        loss = sft_loss(self.policy, batch_tensors(task, pairs), task.prompt_len)
        self.step(loss)
        # report the reward of fresh samples so curves are comparable with RL
        _, rewards, extras, _ = self.rollouts(task, rng, min(32, len(pairs)))
        return {"reward": rewards.mean().item(), "loss": loss.item(), **extras}


# ---------------------------------------------------------------- REINFORCE
class ReinforceLMAgent(LMAgent):
    SPEC = build_spec(
        name="lm_reinforce", label="REINFORCE for LMs", category="language",
        policy_type="on-policy", **TOKEN_SUPPORT,
        description="Sequence-level policy gradient with a batch-mean baseline and a KL "
        "penalty: the simplest RL fine-tuning of a language model.",
        params=_common_params(lr=5e-4) + [KL_PARAM],
        beginner="Generate an answer, check it, and make the whole answer more likely if "
        "it scored above the batch average, less likely if below.",
        technical="∇J ≈ Σ_i (R_i − mean R) ∇ log π_θ(y_i|x_i), the completion's log-"
        "likelihood summed over tokens, plus β·KL(π_θ‖π_ref) per token.",
        intuition="Trial and error on whole answers.",
        update_rule="L = −(R − R̄) Σ_t log π_θ(y_t|x,y_<t) + β KL(π_θ ‖ π_ref)",
        learns="The policy's next-token distribution.",
        failure_modes=["High variance: one baseline for very different prompts",
                       "Without KL the policy can drift into degenerate outputs"],
        expected_metrics=[{"key": "accuracy", "meaning": "Sampled answers that are correct."},
                          {"key": "kl", "meaning": "Divergence from the base model."},
                          {"key": "entropy", "meaning": "Per-token policy entropy; falls as it sharpens."}],
        recommended_envs=["lm_addition", "lm_sort"], related=["grpo", "reinforce"],
    )

    def train_step(self, task, rng):
        seqs, rewards, extras, _ = self.rollouts(task, rng, self.params["prompts_per_step"])
        logp, ent = completion_logprobs(self.policy, seqs, task.prompt_len)
        mask = completion_mask(seqs, task.prompt_len)
        adv = rewards - rewards.mean()
        pg = -(adv * (logp * mask).sum(1)).mean()
        kl = self.kl_to_ref(seqs, task.prompt_len, logp, mask)
        self.step(pg + self.params["kl_coef"] * kl)
        return {"reward": rewards.mean().item(), "kl": kl.item(),
                "entropy": masked_mean(ent, mask).item(), **extras}


# --------------------------------------------------------------------- GRPO
class GRPOAgent(LMAgent):
    SPEC = build_spec(
        name="grpo", label="GRPO", category="language", policy_type="on-policy",
        **TOKEN_SUPPORT,
        description="Group Relative Policy Optimization: several answers per prompt, each "
        "judged against its own group. No value network. Used to train DeepSeek-R1.",
        params=_common_params(lr=5e-4, batch=8) + [
            ParamSpec("group_size", "Group size G", 8, "int",
                      "Completions sampled per prompt.", min=2, max=64),
            KL_PARAM,
            ParamSpec("clip_eps", "Clip ε (lower)", 0.2, "float",
                      "Ratios below 1 − ε are clipped (PPO-style, per token).", min=0.01, max=1.0),
            ParamSpec("clip_eps_high", "Clip ε (upper)", 0.2, "float",
                      "Ratios above 1 + ε are clipped. DAPO's clip-higher raises it (0.28) so "
                      "unlikely-but-good tokens can grow faster, countering entropy collapse.",
                      min=0.01, max=1.0),
            ParamSpec("dynamic_sampling", "Dynamic sampling", False, "bool",
                      "Replace groups whose answers all scored the same (no signal) with fresh "
                      "prompts before updating, as in DAPO."),
            ParamSpec("max_resample_rounds", "Resampling rounds", 3, "int",
                      "How many extra rounds dynamic sampling may draw.", min=1, max=10,
                      advanced=True),
            ParamSpec("epochs", "Epochs per batch", 2, "int",
                      "Gradient passes over each batch of samples.", min=1, max=10),
            ParamSpec("normalize_std", "Divide advantages by group std", True, "bool",
                      "GRPO divides by the group's standard deviation; Dr. GRPO argues this "
                      "biases toward too-easy and too-hard prompts and turns it off."),
            ParamSpec("token_level_loss", "Token-level loss", False, "bool",
                      "Average over all tokens in the batch (DAPO, Dr. GRPO) instead of per "
                      "answer first (GRPO), removing a bias toward short answers."),
        ],
        presets=[
            {"name": "grpo", "label": "GRPO (Shao et al.)",
             "description": "Std-normalised advantages, per-answer loss.", "overrides": {},
             "episodes": 150},
            {"name": "dr-grpo", "label": "Dr. GRPO",
             "description": "No std normalisation, token-level loss.",
             "overrides": {"normalize_std": False, "token_level_loss": True}, "episodes": 150},
            {"name": "dapo", "label": "DAPO",
             "description": "Clip-higher (0.28), dynamic sampling, token-level loss, no KL "
                            "penalty (Yu et al., 2025).",
             "overrides": {"clip_eps_high": 0.28, "dynamic_sampling": True,
                           "token_level_loss": True, "kl_coef": 0.0}, "episodes": 150},
            {"name": "no-kl", "label": "No KL (ablation)",
             "description": "β = 0: nothing anchors the policy to the base model.",
             "overrides": {"kl_coef": 0.0}, "episodes": 150},
        ],
        beginner="For each question, generate eight answers and check them. Answers better "
        "than their siblings become more likely, worse ones less likely. Comparing within the "
        "group replaces the learned critic that PPO would need.",
        technical="For prompt x sample {y_i}_{i=1..G} ~ π_old. Advantage Â_i = (R_i − mean R)/"
        "(std R + ε), shared by all tokens of y_i. Maximise the clipped surrogate "
        "min(ρ_t Â_i, clip(ρ_t, 1±ε) Â_i), ρ_t = π_θ/π_old, minus β·KL(π_θ‖π_ref). Groups where "
        "every answer scores the same carry no signal.",
        intuition="Grade on a curve: what matters is beating your own other attempts.",
        update_rule="Â_i = (R_i − mean_j R_j) / std_j R_j ;  L = −min(ρ Â, clip(ρ, 1±ε) Â) + β KL",
        pseudocode=(
            "for each step:\n"
            "    x = sample prompts;  y_1..y_G ~ π_old(.|x);  R_i = verify(x, y_i)\n"
            "    A_i = (R_i - mean(R)) / std(R)\n"
            "    for epoch in 1..K:\n"
            "        ratio = exp(logπ_θ(y) - logπ_old(y))   # per token\n"
            "        loss = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A) + β*KL(π_θ || π_ref)"
        ),
        learns="The policy only: no value network.",
        failure_modes=["All-correct or all-wrong groups give zero advantage (wasted samples)",
                       "Entropy collapse: the policy stops exploring alternatives",
                       "Std normalisation over-weights prompts that are nearly solved or unsolved"],
        expected_metrics=[{"key": "accuracy", "meaning": "Sampled answers that are correct."},
                          {"key": "kl", "meaning": "Divergence from the base model."},
                          {"key": "entropy", "meaning": "Per-token policy entropy."},
                          {"key": "zero_signal_groups", "meaning": "Share of groups whose answers all scored the same."},
                          {"key": "clip_fraction", "meaning": "Share of tokens whose ratio was clipped."},
                          {"key": "resample_rounds", "meaning": "Extra sampling rounds dynamic sampling needed."}],
        recommended_envs=["lm_addition", "lm_sort", "lm_addition_hackable"],
        related=["ppo", "lm_reinforce", "dpo"],
        paper="Shao et al., 2024 (DeepSeekMath); Guo et al., 2025 (DeepSeek-R1); Yu et al., "
        "2025 (DAPO).",
    )

    def train_step(self, task, rng):
        g = self.params["group_size"]
        want = self.params["prompts_per_step"]
        seqs, rewards, extras, _ = self.rollouts(task, rng, want, g)
        zero_signal = (rewards.view(-1, g).std(1) < 1e-8).float().mean().item()
        rounds = 0
        if self.params["dynamic_sampling"]:
            # keep only groups that disagree; top up with fresh prompts
            def informative(r):
                return r.view(-1, g).std(1) >= 1e-8

            keep = informative(rewards)
            seq_g, rew_g = [seqs.view(-1, g, seqs.size(1))[keep]], [rewards.view(-1, g)[keep]]
            have = int(keep.sum())
            while have < want and rounds < self.params["max_resample_rounds"]:
                rounds += 1
                more_s, more_r, _, _ = self.rollouts(task, rng, want, g)
                k = informative(more_r)
                seq_g.append(more_s.view(-1, g, more_s.size(1))[k])
                rew_g.append(more_r.view(-1, g)[k])
                have += int(k.sum())
            if have == 0:  # nothing to learn from this step
                return {"reward": rewards.mean().item(), "zero_signal_groups": zero_signal,
                        "resample_rounds": float(rounds), **extras}
            seqs = torch.cat(seq_g)[:want].reshape(-1, seqs.size(1))
            rewards = torch.cat(rew_g)[:want].reshape(-1)
        mask = completion_mask(seqs, task.prompt_len)
        grouped = rewards.view(-1, g)
        adv = grouped - grouped.mean(1, keepdim=True)
        if self.params["normalize_std"]:
            adv = adv / (grouped.std(1, keepdim=True) + 1e-4)
        adv = adv.view(-1, 1)
        with torch.no_grad():
            old_logp, _ = completion_logprobs(self.policy, seqs, task.prompt_len)
        lo, hi = self.params["clip_eps"], self.params["clip_eps_high"]
        for _ in range(self.params["epochs"]):
            logp, ent = completion_logprobs(self.policy, seqs, task.prompt_len)
            ratio = torch.exp(logp - old_logp)
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - lo, 1 + hi) * adv)
            if self.params["token_level_loss"]:
                pg = -masked_mean(surrogate, mask)
            else:
                pg = -masked_mean(surrogate, mask, dim=1).mean()
            kl = self.kl_to_ref(seqs, task.prompt_len, logp, mask)
            self.step(pg + self.params["kl_coef"] * kl)
        clipped = masked_mean(((ratio < 1 - lo) | (ratio > 1 + hi)).float(), mask).item()
        return {"reward": rewards.mean().item(), "kl": kl.item(),
                "entropy": masked_mean(ent, mask).item(), "zero_signal_groups": zero_signal,
                "clip_fraction": clipped, "resample_rounds": float(rounds), **extras}


# ---------------------------------------------------------------------- DPO
class DPOAgent(LMAgent):
    SPEC = build_spec(
        name="dpo", label="DPO", category="language", policy_type="offline",
        **TOKEN_SUPPORT,
        description="Direct Preference Optimization: learn from pairs of answers where one "
        "is preferred, without a reward model or RL sampling loop.",
        params=_common_params(lr=5e-4, batch=32) + [
            ParamSpec("beta", "β", 0.1, "float",
                      "Strength of the implicit KL constraint: how far preferences may move "
                      "the policy from the reference.", min=0.001, max=10.0),
            ParamSpec("online", "Online pairs", False, "bool",
                      "Draw pairs from the current policy instead of the reference "
                      "(online / iterative DPO)."),
        ],
        beginner="Show the model two of its own answers and say which is better. It learns "
        "to make the better one relatively more likely than the reference model did.",
        technical="Pairs (y_w, y_l) for prompt x from the reference policy, labelled by the "
        "verifier (ties dropped). Minimise −log σ(β[(log π_θ(y_w) − log π_ref(y_w)) − "
        "(log π_θ(y_l) − log π_ref(y_l))]). The KL-regularised RL optimum, solved in closed "
        "form (Rafailov et al., 2023).",
        intuition="Learning from comparisons rather than scores.",
        update_rule="L = −log σ( β [ Δ_θ(y_w) − Δ_θ(y_l) ] ),  Δ_θ(y) = log π_θ(y|x) − log π_ref(y|x)",
        learns="The policy only: the reward model is implicit.",
        failure_modes=["Offline pairs stop being informative as the policy moves away",
                       "Can lower the likelihood of both answers while widening their gap"],
        expected_metrics=[{"key": "accuracy", "meaning": "Sampled answers that are correct."},
                          {"key": "preference_accuracy", "meaning": "Pairs where the policy prefers the winner."},
                          {"key": "pairs", "meaning": "Usable (non-tied) pairs per step."}],
        recommended_envs=["lm_addition", "lm_sort"], related=["grpo", "lm_sft"],
        paper="Rafailov et al., 2023. Direct Preference Optimization.",
    )

    def train_step(self, task, rng):
        n = self.params["prompts_per_step"]
        pairs = [task.sample(rng) for _ in range(n)]
        prompts = torch.tensor([encode(p) for p, _ in pairs for _ in range(2)])
        source = self.policy if self.params["online"] else self.ref
        source.eval()
        seqs = sample(source, prompts, task.max_new, self.params.get("temperature", 1.0))
        self.policy.train()
        texts = [decode(s[task.prompt_len:]) for s in seqs]
        r = torch.tensor([task.reward(t, a)[0] for t, (_, a) in
                          zip(texts, [p for p in pairs for _ in range(2)])]).view(-1, 2)
        keep = (r[:, 0] != r[:, 1]).nonzero().flatten()
        metrics = {"pairs": float(len(keep))}
        if len(keep):
            idx = torch.stack([keep * 2, keep * 2 + 1], 1)
            win_first = (r[keep, 0] > r[keep, 1])
            w = torch.where(win_first, idx[:, 0], idx[:, 1])
            lo = torch.where(win_first, idx[:, 1], idx[:, 0])
            mask = completion_mask(seqs, task.prompt_len)
            logp, _ = completion_logprobs(self.policy, seqs, task.prompt_len)
            with torch.no_grad():
                ref_logp, _ = completion_logprobs(self.ref, seqs, task.prompt_len)
            delta = ((logp - ref_logp) * mask).sum(1)
            margin = self.params["beta"] * (delta[w] - delta[lo])
            loss = -torch.nn.functional.logsigmoid(margin).mean()
            self.step(loss)
            metrics.update(loss=loss.item(), preference_accuracy=(margin > 0).float().mean().item())
        _, rewards, extras, _ = self.rollouts(task, rng, min(32, n))
        return {"reward": rewards.mean().item(), **metrics, **extras}


LM_AGENTS = {"lm_sft": SFTAgent, "lm_reinforce": ReinforceLMAgent, "grpo": GRPOAgent,
             "dpo": DPOAgent}
