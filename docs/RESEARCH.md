# Research notes: where RL is now, and what the testbed should add

Written October 2026 from a literature search. Each section says what the idea
is, whether the testbed has it, and how it could be learned at small scale.

## 1. RL for language models

**What it is.** Reasoning models are trained by sampling answers from a
pretrained language model, checking them, and reinforcing the good ones.
Group Relative Policy Optimization (GRPO) was introduced by Shao et al. (2024)
and used to train DeepSeek-R1 with verifiable, binary rewards [1]. It drops
PPO's learned critic and instead compares several answers sampled for the same
prompt [1].

**Known problems and fixes.**
- A naive GRPO baseline suffers from entropy collapse, reward noise and
  training instability [2]. DAPO answers this with a decoupled ("clip-higher")
  clipping range, a token-level loss, and by dropping prompts whose sampled
  answers all score the same [2][3].
- Dr. GRPO identifies a response-length bias and a problem-difficulty bias
  caused by GRPO's normalisation terms [3].
- Verifiable rewards resist reward hacking better than learned reward models,
  but are not immune [1].

**In the testbed (v3.5).** The Language-models lab: a ~100k-parameter
transformer, verifiable tasks, SFT, REINFORCE, GRPO (with Dr. GRPO switches)
and DPO, plus a hackable reward that demonstrates reward hacking and the KL
anchor. Entropy collapse is directly visible on GRPO runs.

**Next.** A DAPO preset: asymmetric clipping and skipping zero-signal groups
(the `zero_signal_groups` metric already measures how much signal is wasted).

## 2. World models and model-based RL

**What it is.** An agent learns a model of its environment and uses it.
DreamerV3 learns a world model and trains an actor-critic inside it, with no
planning at decision time [4]. TD-MPC2 instead plans by local trajectory
optimisation in the latent space of a decoder-free model, and outperformed SAC
and DreamerV3 across 104 continuous-control tasks [5].

**In the testbed.** Only tabular Dyna-Q so far.

**Next (in progress).** Two small model-based agents on classic control, where
a model fits in seconds:
- *Planning with a learned model*: an ensemble of small dynamics networks plus
  cross-entropy-method planning (the PETS recipe).
- *Learning in imagination*: an actor-critic trained on rollouts of the learned
  model (Dreamer-style, without pixels).
- A view comparing imagined and real trajectories, so model error is visible.

## 3. Sample-efficient model-free control

CrossQ adds batch normalisation to SAC and reached state-of-the-art sample
efficiency at an update-to-data ratio of 1; adding weight normalisation lets it
scale to higher ratios [6]. SimBa shows that an architecture with observation
normalisation and a residual, layer-normalised network improves many
algorithms simply by being scaled up [7].

**Next.** A network-architecture option (standard MLP vs SimBa-style) shared by
SAC, TD3 and PPO, and CrossQ as an algorithm. Both are small changes with
visible effects on MuJoCo curves.

## 4. Other directions worth a lab or module later

- Offline RL (behaviour cloning, conservative Q-learning) from recorded
  datasets: already on the roadmap.
- Self-play for the board games.
- Exploration bonuses (RND) for sparse rewards.

## Sources

1. Mroueh, *Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss,
   Dynamics, and Success Amplification*, 2025. https://arxiv.org/abs/2503.06639
2. Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*,
   2025. https://arxiv.org/abs/2503.14476
3. *Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal LLMs*, 2025
   (summarises DAPO and Dr. GRPO). https://arxiv.org/abs/2504.21277
4. *Discrete Codebook World Models for Continuous Control*, 2025 (describes
   DreamerV3 and TD-MPC2 as baselines). https://arxiv.org/abs/2503.00653
5. Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous
   Control*, ICLR 2024. https://www.tdmpc2.com
6. Palenicek, Vogt & Peters, *Scaling Off-Policy Reinforcement Learning with
   Batch and Weight Normalization*, 2025. https://arxiv.org/abs/2502.07523
7. Lee et al., *SimBa: Simplicity Bias for Scaling Up Parameters in Deep RL*,
   ICLR 2025. https://arxiv.org/abs/2410.09754
