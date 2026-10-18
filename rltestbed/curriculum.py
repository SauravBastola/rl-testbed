"""The learning path: modules of steps, each one an experiment with a purpose.

A step names the glossary concepts it teaches, carries a ready-to-launch
experiment (the experiment-file format, so it goes through the same validated
launch path as everything else), says what to look for, asks a question with
an answer, and is marked done automatically from the user's runs.

Completion: every run entry in the step's experiment needs at least as many
completed runs of that agent on that environment as it has seeds; sweep entries
need that many completed sweep runs. ``target`` adds a performance bar on the
first entry's best greedy evaluation.
"""

from typing import Any, Dict, List

import yaml


def R(agent, env, episodes, seeds=(0, 1, 2), **extra):
    return {"agent": agent, "env": env, "episodes": episodes, "seeds": list(seeds), **extra}


MODULES: List[Dict[str, Any]] = [
    {
        "id": "first-contact",
        "title": "First contact",
        "summary": "What an agent, an episode and a learning curve are, and what learning looks like.",
        "steps": [
            {
                "id": "random-baseline",
                "title": "Measure the floor",
                "summary": "Before training anything, see what no skill looks like.",
                "concepts": ["agent", "episode", "return", "policy"],
                "experiment": {"runs": [R("random", "gridworld", 50, seeds=(0,))]},
                "watch": "Episode length jumps around and often hits the 200-step limit; return "
                         "is mostly the small per-step penalty piling up.",
                "question": "Why is a random agent's return in Gridworld negative rather than zero?",
                "answer": "Every step costs a little, and a random walk takes many steps to "
                          "stumble on the goal, so the penalties add up before the episode ends.",
            },
            {
                "id": "first-learning",
                "title": "Watch values spread",
                "summary": "Tabular Q-learning on the same world, three seeds.",
                "concepts": ["q-function", "td-learning", "td-error", "bootstrapping", "greedy"],
                "experiment": {"runs": [R("q_learning", "gridworld", 300)]},
                "watch": "Episode length falls from about 50 toward the optimal 10 steps, and the "
                         "greedy evaluations reach the best return well before training returns do.",
                "question": "Why do greedy evaluations look better than the training curve?",
                "answer": "Training episodes still explore (ε-greedy), so they include deliberate "
                          "random moves; evaluations act greedily on what has been learned.",
                "target": {"metric": "best_eval", "at_least": 0.9},
            },
        ],
    },
    {
        "id": "exploration",
        "title": "Exploration",
        "summary": "How to choose between what you know and what you could learn, isolated in bandits.",
        "steps": [
            {
                "id": "ucb-vs-greedy",
                "title": "Directed versus random exploration",
                "summary": "UCB against ε-greedy sample averages on a 10-armed bandit.",
                "concepts": ["exploration", "epsilon-greedy", "ucb", "regret"],
                "experiment": {"runs": [
                    R("ucb", "multi_armed_bandit", 1000, name="UCB"),
                    R("q_learning", "multi_armed_bandit", 1000, name="epsilon-greedy",
                      params={"gamma": 0.0, "alpha": 0.1}),
                ]},
                "watch": "Compare the mean-return curves with ±1σ bands: UCB tends to climb "
                         "faster and keep rising while ε-greedy flattens.",
                "question": "ε-greedy keeps exploring at rate ε forever. What does that cost on a "
                            "stationary bandit?",
                "answer": "Once the best arm is known, every exploratory pull is a guaranteed loss, "
                          "so regret keeps growing linearly; UCB's bonus shrinks with evidence.",
            },
            {
                "id": "drifting",
                "title": "When the world changes",
                "summary": "Sample averages versus a constant step size on a drifting bandit.",
                "concepts": ["nonstationary", "step-size"],
                "experiment": {"runs": [
                    R("ucb", "nonstationary_bandit", 2000, name="UCB, sample averages"),
                    R("ucb", "nonstationary_bandit", 2000, name="UCB, step size 0.1",
                      params={"alpha": 0.1}),
                ]},
                "watch": "The constant step size keeps up with the drift; sample averages fall "
                         "behind as old pulls dominate their estimates.",
                "question": "Why does averaging all past rewards fail here?",
                "answer": "The true arm values drift, so old rewards describe a world that no "
                          "longer exists; a constant step size weights recent rewards more.",
            },
        ],
    },
    {
        "id": "tabular-control",
        "title": "Learning to act",
        "summary": "On-policy versus off-policy learning, and planning with a model.",
        "steps": [
            {
                "id": "cliff",
                "title": "SARSA versus Q-learning on the cliff",
                "summary": "The classic on-policy/off-policy comparison.",
                "concepts": ["on-policy", "off-policy", "epsilon-greedy"],
                "experiment": {"runs": [
                    R("sarsa", "cliff_walking", 400, name="SARSA"),
                    R("q_learning", "cliff_walking", 400, name="Q-learning"),
                ]},
                "watch": "Training return: SARSA is higher. Greedy evaluation: Q-learning reaches "
                         "the optimal -13 while SARSA settles on a longer, safer path.",
                "question": "Q-learning learns the best path, yet earns less while training. Why?",
                "answer": "It learns the value of the greedy policy, which walks the cliff edge, "
                          "but it acts ε-greedily, so random steps there sometimes fall off. SARSA "
                          "learns the value of the policy it actually follows, and so avoids the edge.",
            },
            {
                "id": "dyna",
                "title": "Plan with a model",
                "summary": "Dyna-Q with planning against plain Q-learning on the Dyna maze.",
                "concepts": ["model", "planning", "dyna", "sample-efficiency", "sparse-reward"],
                "experiment": {"runs": [
                    R("dyna_q", "dyna_maze", 50, name="Dyna-Q (10 planning steps)"),
                    R("q_learning", "dyna_maze", 50, name="Q-learning"),
                ]},
                "watch": "Steps per episode: Dyna-Q drops to about 14 within a few episodes; "
                         "Q-learning needs many more.",
                "question": "Both agents receive exactly the same real experience. Where does "
                            "Dyna-Q's advantage come from?",
                "answer": "Each real step is replayed in imagination many times through the learned "
                          "model, so the single reward at the goal propagates back through the maze "
                          "far faster than one update per real step allows.",
            },
        ],
    },
    {
        "id": "deep-value",
        "title": "Deep value-based RL",
        "summary": "From tables to networks: DQN and the tricks that make it stable.",
        "steps": [
            {
                "id": "probe-first",
                "title": "Check the machinery first",
                "summary": "DQN on a two-step probe with a delayed reward.",
                "concepts": ["probe", "credit-assignment", "target-network"],
                "experiment": {"runs": [R("dqn", "probe3", 300, seeds=(0,),
                                          params={"target_update_freq": 20})]},
                "watch": "Greedy evaluation should reach 1.0: the agent must value a first step "
                         "that pays nothing because of what it leads to.",
                "question": "Why test on a two-state toy before CartPole?",
                "answer": "It has a known answer and trains in seconds. If DQN fails here the "
                          "cause is a bug or a setting, not the difficulty of the task.",
                "target": {"metric": "best_eval", "at_least": 0.95},
            },
            {
                "id": "dqn-cartpole",
                "title": "DQN balances a pole",
                "summary": "The canonical first deep-RL result, over three seeds.",
                "concepts": ["function-approximation", "replay-buffer", "target-network", "threshold"],
                "experiment": {"runs": [R("dqn", "cartpole", 300)]},
                "watch": "Greedy evaluations climb toward the solved line at 475 well before the "
                         "noisy training curve does, and seeds differ a lot.",
                "question": "Why does DQN need a replay buffer when tabular Q-learning did not?",
                "answer": "Consecutive steps are strongly correlated, and a network trained on them "
                          "in order overfits to the latest experience. Sampling old transitions at "
                          "random decorrelates updates and reuses data.",
            },
            {
                "id": "lr-sweep",
                "title": "How sensitive is it?",
                "summary": "A learning-rate sweep for DQN on CartPole.",
                "concepts": ["hyperparameter", "sweep", "step-size"],
                "experiment": {"sweeps": [{
                    "name": "DQN learning rate", "agent": "dqn", "env": "cartpole",
                    "episodes": 300, "seeds": [0, 1], "strategy": "grid",
                    "space": {"lr": {"values": [0.0001, 0.0005, 0.002]}},
                }]},
                "watch": "On the sweep page, check the effect panel for lr and whether the ranking "
                         "is clear or the seeds overlap.",
                "question": "With two seeds per setting, how confident can you be in the winner?",
                "answer": "Not very, unless the gap is much larger than the spread between seeds. "
                          "Rerun the top two settings with more seeds before concluding.",
            },
            {
                "id": "dqn-variants",
                "title": "Fixing overestimation",
                "summary": "DQN, Double DQN and Dueling DQN on CartPole.",
                "concepts": ["overestimation", "dueling", "variance"],
                "experiment": {"runs": [
                    R("dqn", "cartpole", 300, seeds=(0, 1), name="DQN"),
                    R("dqn_double", "cartpole", 300, seeds=(0, 1), name="Double DQN"),
                    R("dueling_dqn", "cartpole", 300, seeds=(0, 1), name="Dueling DQN"),
                ]},
                "watch": "Differences on CartPole are often within the seed bands. That is itself "
                         "the lesson: improvements shown on hard benchmarks need not show on easy ones.",
                "question": "Why would taking a max over noisy estimates bias them upward?",
                "answer": "The max picks whichever estimate happens to be most over-estimated, so "
                          "its expected value exceeds the true maximum. Double DQN evaluates the "
                          "chosen action with a different network to break that link.",
            },
        ],
    },
    {
        "id": "policy-gradient",
        "title": "Policy gradients",
        "summary": "Learning the policy directly, and the variance problem that shapes every method.",
        "steps": [
            {
                "id": "baseline",
                "title": "Taming variance with a baseline",
                "summary": "REINFORCE with and without a learned baseline.",
                "concepts": ["policy-gradient", "reinforce", "baseline", "variance"],
                "experiment": {"runs": [
                    R("reinforce", "cartpole", 500, name="REINFORCE"),
                    R("reinforce_baseline", "cartpole", 500, name="REINFORCE + baseline"),
                ]},
                "watch": "Compare the width of the ±1σ bands as much as the means.",
                "question": "Subtracting a baseline changes every update. Why does it not bias the gradient?",
                "answer": "The baseline does not depend on the action, and the expected gradient of "
                          "log π over actions is zero, so the subtracted term averages out. Only the "
                          "variance changes.",
            },
            {
                "id": "actor-critic",
                "title": "Actor-critic and trust regions",
                "summary": "A2C against PPO on CartPole.",
                "concepts": ["actor-critic", "advantage", "gae", "ppo-clip", "entropy", "kl"],
                "experiment": {"runs": [
                    R("a2c", "cartpole", 300, name="A2C"),
                    R("ppo", "cartpole", 300, name="PPO", params={"rollout_steps": 512}),
                ]},
                "watch": "On a PPO run, open the metrics: entropy should fall gently and approx_kl "
                         "stay small. Spikes in KL often precede collapses.",
                "question": "What would happen to PPO without clipping, running ten epochs on each rollout?",
                "answer": "It would keep pushing the policy in directions that looked good on stale "
                          "data, often far beyond where the advantage estimates are valid, and "
                          "performance can collapse. Clipping removes the incentive past 1 ± ε.",
            },
        ],
    },
    {
        "id": "continuous",
        "title": "Continuous control",
        "summary": "Actions as vectors of torques instead of buttons.",
        "steps": [
            {
                "id": "pendulum",
                "title": "Swing up a pendulum",
                "summary": "SAC and TD3 on Pendulum, the smallest continuous task.",
                "concepts": ["gaussian-policy", "dpg", "max-entropy", "squashing"],
                "experiment": {"runs": [
                    R("sac", "pendulum", 100, seeds=(0, 1), name="SAC"),
                    R("td3", "pendulum", 100, seeds=(0, 1), name="TD3"),
                ]},
                "watch": "Returns start near -1200 (flailing) and should rise toward -200 as the "
                         "pendulum learns to swing up and balance. Watch SAC's alpha shrink.",
                "question": "TD3's actor is deterministic. How does it explore?",
                "answer": "By adding noise to its actions during training only; evaluation uses "
                          "the noiseless action. SAC instead explores through its stochastic policy "
                          "and the entropy bonus.",
            },
            {
                "id": "crossq",
                "title": "Stability without target networks",
                "summary": "SAC against CrossQ on Pendulum.",
                "concepts": ["crossq", "target-network", "obs-normalization", "truncation"],
                "experiment": {"runs": [
                    R("sac", "pendulum", 40, seeds=(0, 1), name="SAC"),
                    R("crossq", "pendulum", 40, seeds=(0, 1), name="CrossQ"),
                ]},
                "watch": "Both should reach about -150 within 40 episodes. Compare how early each "
                         "starts improving, and the time per episode: CrossQ's critics are wider.",
                "question": "A target network makes learning stable by lagging behind. How does CrossQ "
                            "stay stable without one?",
                "answer": "It batch-normalises its critics and passes current and next state-action "
                          "pairs through them in one batch, so both halves of the TD target are "
                          "normalised by the same statistics and stay consistent as the critic changes.",
            },
            {
                "id": "mujoco",
                "title": "A simulated body",
                "summary": "SAC on MuJoCo's inverted pendulum. Needs the MuJoCo extra.",
                "concepts": ["obs-normalization", "sample-efficiency"],
                "experiment": {"runs": [R("sac", "inverted_pendulum", 100, seeds=(0,))]},
                "watch": "Watch it in Visualize afterwards: the observation panel names every joint.",
                "question": "Why normalise observations for MuJoCo but not for Gridworld?",
                "answer": "MuJoCo observations mix quantities on very different scales (angles near "
                          "1, velocities near 10). Gridworld's one-hot input is already on one scale.",
            },
        ],
    },
    {
        "id": "world-models",
        "title": "World models",
        "summary": "Learn how the world works, then plan or practise inside the model.",
        "steps": [
            {
                "id": "wm-planning",
                "title": "Plan with a learned model",
                "summary": "PETS against SAC on Pendulum, with the same small budget of episodes.",
                "concepts": ["world-model", "mpc", "cem", "sample-efficiency"],
                "experiment": {"runs": [
                    R("pets", "pendulum", 12, seeds=(0, 1), name="PETS"),
                    R("sac", "pendulum", 12, seeds=(0, 1), name="SAC"),
                ]},
                "watch": "PETS swings the pendulum up within about five episodes; SAC is still "
                         "learning at twelve. Compare wall-clock time too: PETS plans at every step.",
                "question": "PETS never trains a policy. How does it choose actions?",
                "answer": "At every step it imagines hundreds of action sequences in its learned model, "
                          "scores each by imagined return, refines them with the cross-entropy method, "
                          "executes the first action of the best one, and then plans again.",
            },
            {
                "id": "wm-compounding",
                "title": "Where imagination breaks",
                "summary": "PETS on CartPole; open the Imagination tab of the run.",
                "concepts": ["compounding-error", "epistemic"],
                "experiment": {"runs": [R("pets", "cartpole", 10, seeds=(0,))]},
                "watch": "Compare model_error_1 with model_error_15 in the metric charts, then look at "
                         "how the imagined trajectory leaves the real one after a few steps.",
                "question": "The model is nearly exact one step ahead. Why does it drift over fifteen?",
                "answer": "Each prediction becomes the input to the next, so small errors are fed back "
                          "and grow, often into states the model never saw in training. Short horizons "
                          "and replanning every step keep this under control.",
            },
            {
                "id": "wm-imagination",
                "title": "Learning in imagination",
                "summary": "Dreamer-lite trains a policy inside the model; compare it with planning.",
                "concepts": ["imagination", "actor-critic", "mpc"],
                "experiment": {"runs": [
                    R("dreamer_lite", "pendulum", 20, seeds=(0,), name="Dreamer-lite"),
                    R("pets", "pendulum", 20, seeds=(0,), name="PETS"),
                ]},
                "watch": "Both learn within a handful of episodes. Dreamer-lite acts with one network "
                         "pass per step; PETS plans every step. Its imagined_return metric shows what "
                         "the policy expects; compare it with real returns.",
                "question": "Why would imagined returns be higher than real ones?",
                "answer": "The actor is optimised against the model, so it seeks out actions where the "
                          "model is too optimistic. Errors in the world model become errors in the "
                          "policy, which is why imagined rollouts are kept short.",
            },
        ],
    },
    {
        "id": "language-models",
        "title": "RL for language models",
        "summary": "The methods behind reasoning models, on a tiny transformer: verifiers, groups, KL and preferences.",
        "steps": [
            {
                "id": "lm-verifier",
                "title": "Sharpen a model with a verifier",
                "summary": "GRPO and REINFORCE fine-tune a pretrained model on addition, scored right or wrong.",
                "concepts": ["pretraining", "rlvr", "grpo-term", "sharpening", "reference-model"],
                "experiment": {"runs": [
                    R("grpo", "lm_addition", 60, seeds=(0, 1), name="GRPO"),
                    R("lm_reinforce", "lm_addition", 60, seeds=(0, 1), name="REINFORCE"),
                ]},
                "watch": "Sampled accuracy rises from about 0.5 toward 0.9 while entropy falls and KL "
                         "to the base model grows. Open the Samples tab of a run to read its answers.",
                "question": "The base model's greedy answer is already right almost every time. What "
                            "is RL improving?",
                "answer": "The sampling distribution: a single sample at temperature 1 is right only "
                          "about half the time. RL moves probability onto the correct answer, so samples "
                          "agree with the greedy answer. This sharpening is most of what RL with "
                          "verifiable rewards does.",
            },
            {
                "id": "lm-hacking",
                "title": "Reward hacking and the KL anchor",
                "summary": "A flawed reward model pays for length. Compare no KL penalty with a strong one.",
                "concepts": ["reward-model", "reward-hacking", "reference-model", "kl"],
                "experiment": {"runs": [
                    R("grpo", "lm_addition_hackable", 80, seeds=(0, 1), name="GRPO, no KL",
                      params={"kl_coef": 0.0}),
                    R("grpo", "lm_addition_hackable", 80, seeds=(0, 1), name="GRPO, KL 0.5",
                      params={"kl_coef": 0.5}),
                ]},
                "watch": "Switch the comparison metric between reward and accuracy. Without KL, reward "
                         "climbs to its maximum of 2.0 while accuracy falls to zero; with KL 0.5, "
                         "reward rises less and accuracy rises.",
                "question": "Reward went up and the model got worse. How would you notice this if you "
                            "only had the reward?",
                "answer": "You could not, from the reward alone. You need a second measurement the "
                          "policy is not optimising (here, accuracy), samples you read yourself, or a "
                          "KL budget that flags runaway drift. Measuring only the optimised quantity "
                          "hides reward hacking.",
            },
            {
                "id": "lm-dapo",
                "title": "Refining GRPO: DAPO",
                "summary": "GRPO against its DAPO variant on addition.",
                "concepts": ["dapo-term", "entropy", "grpo-term"],
                "experiment": {"runs": [
                    R("grpo", "lm_addition", 60, seeds=(0, 1), name="GRPO"),
                    R("grpo", "lm_addition", 60, seeds=(0, 1), name="DAPO",
                      params={"clip_eps_high": 0.28, "dynamic_sampling": True,
                              "token_level_loss": True, "kl_coef": 0.0}),
                ]},
                "watch": "DAPO tends to be ahead early (dynamic sampling means every update carries "
                         "signal) and needs extra sampling rounds to do it. Compare the entropy charts: "
                         "here they fall about equally.",
                "question": "Clip-higher exists to stop entropy collapsing. Why is the effect hard to "
                            "see in this lab?",
                "answer": "Collapse becomes a problem over long training on hard problems, where the "
                          "policy must keep exploring for many steps. These tasks are learned in a few "
                          "dozen steps and the right answer is already likely, so there is little "
                          "exploration left to protect. Results at small scale do not always transfer.",
            },
            {
                "id": "lm-preferences",
                "title": "Preferences instead of rewards",
                "summary": "DPO learns from which of two answers is better; GRPO from verified scores.",
                "concepts": ["dpo-term", "reference-model", "pretraining"],
                "experiment": {"runs": [
                    R("dpo", "lm_sort", 60, seeds=(0, 1), name="DPO"),
                    R("grpo", "lm_sort", 60, seeds=(0, 1), name="GRPO"),
                ]},
                "watch": "DPO improves less than GRPO here: its pairs come from the fixed base model, "
                         "so they stop being informative as the policy moves. Try the online-pairs option.",
                "question": "DPO never computes a reward during training. Where does the information "
                            "come from?",
                "answer": "From the preference labels on each pair: which answer is better. DPO turns "
                          "the KL-regularised RL objective into a classification loss on those pairs, "
                          "measured relative to the reference model.",
            },
        ],
    },
    {
        "id": "science",
        "title": "Doing RL science",
        "summary": "Reading results honestly: seeds, spread and what counts as evidence.",
        "steps": [
            {
                "id": "seeds",
                "title": "One run is an anecdote",
                "summary": "PPO on CartPole over five seeds.",
                "concepts": ["seed", "variance", "confidence-band", "learning-curve", "evaluation"],
                "experiment": {"runs": [R("ppo", "cartpole", 300, seeds=(0, 1, 2, 3, 4),
                                          params={"rollout_steps": 512})]},
                "watch": "Open the comparison and switch between pooled seeds and each run. Note "
                         "how different the best and worst seeds look.",
                "question": "If you had run only the luckiest seed, what would you have concluded?",
                "answer": "That PPO solves CartPole quickly and reliably, which overstates it. "
                          "Report the mean and spread across seeds, chosen before looking at results.",
            },
        ],
    },
]


def iter_steps():
    for module in MODULES:
        for step in module["steps"]:
            yield module, step


def experiment_text(step: Dict[str, Any]) -> str:
    """The step's experiment as an experiment file, ready to launch or edit."""
    doc = {"version": 1, "name": step["title"], "description": step["summary"]}
    doc.update({k: v for k, v in step["experiment"].items() if v})
    return ("# Learning path step: " + step["title"] + "\n"
            + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))


def _entries(step):
    exp = step["experiment"]
    for run in exp.get("runs", []):
        yield "run", run
    for sweep in exp.get("sweeps", []):
        yield "sweep", sweep


def step_status(step: Dict[str, Any], runs: List[Dict[str, Any]],
                envs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """done / running / todo / unavailable, with the runs that count toward it."""
    missing = sorted({e["env"] for _, e in _entries(step)
                      if envs.get(e["env"], {}).get("available") is False})
    matched, satisfied, active = [], True, False
    for kind, entry in _entries(step):
        mine = [r for r in runs if r["agent"] == entry["agent"] and r["env"] == entry["env"]
                and (kind == "run" or "sweep" in (r.get("tags") or []))
                and all((r.get("params") or {}).get(k) == v
                        for k, v in (entry.get("params") or {}).items())]
        done = [r for r in mine if r["status"] == "completed"]
        need = len(entry.get("seeds") or [0])
        if kind == "sweep":
            need *= max(1, len(next(iter(entry["space"].values())).get("values", [1])))
        satisfied &= len(done) >= need
        active |= any(r["status"] in ("queued", "created", "running", "stopping") for r in mine)
        matched += [r["id"] for r in mine]
    target = step.get("target")
    best = None
    if target:
        first = next(iter(_entries(step)))[1]
        evals = [r.get("best_eval_reward") for r in runs
                 if r["agent"] == first["agent"] and r["env"] == first["env"]
                 and r.get("best_eval_reward") is not None]
        best = max(evals) if evals else None
        satisfied &= best is not None and best >= target["at_least"]
    state = ("unavailable" if missing else "done" if satisfied
             else "running" if active else "todo")
    return {"state": state, "run_ids": matched[-60:], "missing_envs": missing, "best_eval": best}


def with_status(runs: List[Dict[str, Any]], envs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    modules = []
    n_done = n_total = 0
    for module in MODULES:
        steps = []
        for step in module["steps"]:
            status = step_status(step, runs, envs)
            n_total += 1
            n_done += status["state"] == "done"
            steps.append({**step, "status": status, "experiment_text": experiment_text(step)})
        modules.append({**module, "steps": steps,
                        "done": sum(s["status"]["state"] == "done" for s in steps)})
    return {"modules": modules, "done": n_done, "total": n_total}


def legacy_path() -> List[Dict[str, Any]]:
    """The v3 /api/learning-path shape, derived from the curriculum."""
    out = []
    for i, (_, step) in enumerate(iter_steps(), start=1):
        _, first = next(iter(_entries(step)))
        out.append({"step": i, "title": step["title"], "env": first["env"],
                    "algorithm": first["agent"], "why": step["summary"],
                    "done_when": step["watch"]})
    return out
