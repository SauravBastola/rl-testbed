"""The glossary: every term the testbed uses, at two levels of detail.

Each entry has a one-line ``definition`` (kept for the v3 API shape), a longer
``detail``, an optional ``formula`` (Markdown with $LaTeX$), ``related`` term
ids, and the ``algorithms`` / ``envs`` where the idea can be seen in practice.
``tests/test_learning.py`` checks that every reference resolves.
"""

from typing import Any, Dict, List

CATEGORIES = [
    "Basics",
    "Learning from experience",
    "Deep value methods",
    "Policy methods",
    "Continuous control",
    "Planning",
    "Doing experiments",
]


def T(id, term, category, definition, detail, formula="", related=(), algorithms=(), envs=()):
    return {"id": id, "term": term, "category": category, "definition": definition,
            "detail": detail, "formula": formula, "related": list(related),
            "algorithms": list(algorithms), "envs": list(envs)}


GLOSSARY: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ Basics
    T("agent", "Agent", "Basics",
      "The learner: it observes, chooses actions, and improves from the rewards it gets.",
      "In the testbed an agent is one algorithm with its own parameters and memory. It sees "
      "only observations and rewards, never the environment's internals.",
      related=["environment", "policy"]),
    T("environment", "Environment", "Basics",
      "The world the agent acts in: it returns the next observation and a reward for every action.",
      "Each environment defines what can be observed, which actions exist, how rewards are "
      "paid, and when an episode ends. The testbed groups environments into labs by the shape "
      "of the problem they pose.",
      related=["agent", "mdp", "episode"], envs=["gridworld", "cartpole"]),
    T("mdp", "Markov decision process (MDP)", "Basics",
      "The mathematical model behind most RL: states, actions, transition probabilities, rewards and a discount.",
      "\"Markov\" means the next state depends only on the current state and action, not on the "
      "history. When the agent cannot see the full state (as in poker), the problem is partially "
      "observable and memory starts to matter.",
      formula=r"$(\mathcal{S}, \mathcal{A}, p(s', r \mid s, a), \gamma)$",
      related=["state", "action", "reward", "discount"]),
    T("state", "State and observation", "Basics",
      "The state is the full situation of the environment; the observation is what the agent gets to see of it.",
      "In Gridworld they coincide: the agent knows its cell. In MuJoCo the observation is a vector of "
      "joint angles and velocities. In Texas Hold'em the opponent's cards are part of the state but "
      "not of the observation. The Visualize page shows exactly what the agent observes.",
      related=["mdp", "one-hot"], envs=["gridworld", "half_cheetah"]),
    T("one-hot", "One-hot encoding", "Basics",
      "A vector of zeros with a single 1 marking which of N discrete states (or categories) applies.",
      "Tabular environments hand the agent its state this way. It carries no notion of distance: "
      "state 3 is no more similar to state 4 than to state 40, so nothing learned in one state "
      "transfers to its neighbours.",
      related=["state", "function-approximation"], envs=["frozen_lake", "dyna_maze"]),
    T("action", "Action", "Basics",
      "What the agent does at each step: a choice from a list (discrete) or a vector of numbers (continuous).",
      "CartPole has two discrete actions, push left or push right. HalfCheetah's action is six "
      "joint torques between -1 and 1. Algorithms are built for one kind or the other, which is "
      "why the testbed checks compatibility before a run starts.",
      related=["action-mask", "gaussian-policy"], envs=["cartpole", "half_cheetah"]),
    T("action-mask", "Action mask", "Basics",
      "A list of which actions are legal right now; mask-aware algorithms never choose an illegal one.",
      "In Connect Four a full column cannot take another disc. The environment sends a mask with "
      "every observation, and the agent sets the probability (or value) of illegal actions to "
      "nothing before choosing.",
      related=["action"], envs=["tictactoe", "connect_four"]),
    T("reward", "Reward", "Basics",
      "A number the environment returns after each action; the agent's only signal of what is good.",
      "Rewards define the task. CartPole pays +1 for every step the pole stays up; CliffWalking "
      "charges -1 per step and -100 for falling. Changing the reward changes what is optimal.",
      related=["return", "sparse-reward"]),
    T("episode", "Episode", "Basics",
      "One run of the environment from reset to an end: termination (the task ended) or truncation (a time limit).",
      "Learning curves plot one point per episode. The distinction matters for learning: at "
      "termination the future is worth nothing, while at truncation the agent was simply cut off.",
      related=["return", "trajectory"]),
    T("trajectory", "Trajectory", "Basics",
      "The sequence of states, actions and rewards produced by acting in an environment.",
      "Visualize plays one trajectory step by step. Monte Carlo methods learn from whole "
      "trajectories; temporal-difference methods learn from single steps of them.",
      formula=r"$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$",
      related=["episode", "monte-carlo", "td-learning"]),
    T("return", "Return", "Basics",
      "The total (discounted) reward from a time step to the end of the episode: what the agent tries to maximise.",
      "Learning curves plot the return of each training episode. Because exploration adds noise, "
      "the testbed also plots a greedy evaluation every few episodes.",
      formula=r"$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$",
      related=["discount", "reward", "evaluation"]),
    T("discount", "Discount factor", "Basics",
      "γ in [0, 1): how much a reward one step later is worth compared with one now.",
      "It sets the agent's effective planning horizon, roughly $1/(1-\\gamma)$ steps: 0.99 looks "
      "about 100 steps ahead, 0.9 about 10. Too low and long-term goals become invisible; too high "
      "and learning slows and becomes noisier.",
      formula=r"$\text{horizon} \approx \frac{1}{1-\gamma}$",
      related=["return"]),
    T("policy", "Policy", "Basics",
      "The agent's behaviour: a rule that maps each observation to an action, or to probabilities over actions.",
      "A deterministic policy picks one action per state; a stochastic policy samples from a "
      "distribution. Policy-gradient methods learn the policy directly; value-based methods derive "
      "it from learned values by acting greedily.",
      formula=r"$\pi(a \mid s)$",
      related=["value-function", "greedy", "policy-gradient"]),
    T("value-function", "Value function", "Basics",
      "V(s): the return the agent can expect from state s if it follows its policy from there.",
      "Values turn a sparse, delayed reward into a signal available in every state. A critic in "
      "actor-critic methods is a learned value function.",
      formula=r"$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right]$",
      related=["q-function", "bellman", "actor-critic"]),
    T("q-function", "Action value (Q-function)", "Basics",
      "Q(s, a): the expected return from taking action a in state s and following the policy afterwards.",
      "Knowing Q makes acting easy: pick the action with the highest value. Q-learning, SARSA and "
      "DQN all learn Q; tabular methods store it in a table, DQN approximates it with a network.",
      formula=r"$Q^\pi(s,a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$",
      related=["value-function", "advantage", "greedy"], algorithms=["q_learning", "dqn"]),
    T("bellman", "Bellman equation", "Basics",
      "The recursive relation that values satisfy: a state's value is the immediate reward plus the discounted value of what follows.",
      "Nearly every value-based method is a way of making its estimates satisfy this equation. "
      "The optimality version, with a max over next actions, is what Q-learning chases.",
      formula=r"$Q^*(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$",
      related=["td-learning", "bootstrapping", "q-function"], algorithms=["q_learning"]),
    T("greedy", "Greedy action", "Basics",
      "The action with the highest estimated value right now.",
      "Always acting greedily never tests whether another action is better, so learners mix in "
      "exploration during training. Evaluations in the testbed use greedy actions to measure what "
      "has been learned without that noise.",
      formula=r"$a = \arg\max_{a} Q(s, a)$",
      related=["exploration", "epsilon-greedy", "evaluation"]),

    # ------------------------------------------------ Learning from experience
    T("exploration", "Exploration vs exploitation", "Learning from experience",
      "The trade-off between trying actions to learn about them and choosing the action that currently looks best.",
      "Too little exploration and the agent settles on the first decent behaviour it finds; too "
      "much and it never benefits from what it knows. Bandits isolate this trade-off.",
      related=["epsilon-greedy", "ucb", "thompson", "regret"], envs=["multi_armed_bandit"]),
    T("epsilon-greedy", "ε-greedy", "Learning from experience",
      "With probability ε take a random action, otherwise the greedy one. ε usually decays during training.",
      "Simple and effective, but blind: it explores a hopeless action as often as a promising one. "
      "Watch the epsilon metric on a run to see the schedule.",
      formula=r"$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & a = \arg\max Q(s,\cdot) \\ \frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$",
      related=["exploration", "greedy"], algorithms=["q_learning", "dqn"]),
    T("ucb", "Upper confidence bound (UCB)", "Learning from experience",
      "Explore by optimism: add a bonus to each action's value that shrinks the more often it has been tried.",
      "Rarely tried actions look attractive until there is evidence about them, so exploration is "
      "directed rather than random, and there is no ε to tune.",
      formula=r"$a_t = \arg\max_a \left[ Q(a) + c\sqrt{\frac{\ln t}{N(a)}} \right]$",
      related=["exploration", "regret", "thompson"], algorithms=["ucb"], envs=["multi_armed_bandit"]),
    T("thompson", "Thompson sampling", "Learning from experience",
      "Keep a probability distribution over each action's value, sample one value per action, and act greedily on the samples.",
      "Each action is chosen with roughly the probability that it is the best one. Uncertain "
      "actions sometimes draw high samples and get tried; as evidence grows they stop.",
      related=["exploration", "ucb"], algorithms=["thompson"], envs=["multi_armed_bandit"]),
    T("regret", "Regret", "Learning from experience",
      "The reward lost by not always choosing the best action: the standard measure of exploration quality.",
      "A good exploration strategy has regret that grows ever more slowly. Comparing cumulative "
      "return of UCB and ε-greedy on a bandit is a direct look at regret.",
      formula=r"$L_T = T\mu^* - \sum_{t=1}^{T} \mathbb{E}[r_t]$",
      related=["exploration", "ucb"]),
    T("nonstationary", "Nonstationarity", "Learning from experience",
      "The environment changes over time, so old experience becomes misleading.",
      "Sample averages weigh all of history equally and fall behind; a constant step size weighs "
      "recent experience more. The Drifting Bandit makes the difference visible.",
      formula=r"$Q \leftarrow Q + \alpha\,(r - Q)$",
      related=["step-size"], envs=["nonstationary_bandit"]),
    T("step-size", "Step size (learning rate)", "Learning from experience",
      "α: how far each update moves an estimate toward its target.",
      "Large steps learn fast but noisily and can diverge; small steps are stable but slow. In deep "
      "RL the learning rate of the optimiser plays this role and is often the most sensitive "
      "hyperparameter, which is what sweeps are for.",
      related=["hyperparameter", "sweep", "nonstationary"]),
    T("monte-carlo", "Monte Carlo learning", "Learning from experience",
      "Learn from complete episodes: move each estimate toward the return that actually followed.",
      "Unbiased, since it uses real returns, but high variance, and nothing is learned until an "
      "episode ends.",
      formula=r"$V(s_t) \leftarrow V(s_t) + \alpha\left[G_t - V(s_t)\right]$",
      related=["td-learning", "return"], algorithms=["mc_control", "reinforce"]),
    T("td-learning", "Temporal-difference (TD) learning", "Learning from experience",
      "Learn from single steps: move an estimate toward the reward plus the estimated value of the next state.",
      "TD methods learn during the episode and usually faster than Monte Carlo, at the price of "
      "some bias from trusting their own estimates.",
      formula=r"$V(s_t) \leftarrow V(s_t) + \alpha\left[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\right]$",
      related=["td-error", "bootstrapping", "monte-carlo"], algorithms=["q_learning", "sarsa"]),
    T("td-error", "TD error", "Learning from experience",
      "δ: the gap between an estimate and its one-step target. Most value methods try to shrink it.",
      "A TD error that never falls points to a bug or too high a learning rate. Tabular agents "
      "report it as the td_error metric.",
      formula=r"$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$",
      related=["td-learning", "bootstrapping"], algorithms=["q_learning"]),
    T("bootstrapping", "Bootstrapping", "Learning from experience",
      "Updating an estimate from another estimate, instead of waiting for the true return.",
      "It is what lets TD methods learn mid-episode, and also what lets errors feed on themselves: "
      "a wrong value of the next state becomes a wrong target for this one.",
      related=["td-learning", "target-network", "overestimation"]),
    T("on-policy", "On-policy learning", "Learning from experience",
      "Learn about the policy you are currently following, so data must come from that policy.",
      "SARSA, REINFORCE, A2C and PPO are on-policy. They cannot reuse old experience after the "
      "policy changes, which makes them less sample-efficient but often more stable.",
      related=["off-policy", "replay-buffer"], algorithms=["sarsa", "ppo"]),
    T("off-policy", "Off-policy learning", "Learning from experience",
      "Learn about one policy (usually the greedy one) from data collected by another (usually an exploring one).",
      "Q-learning, DQN, SAC and TD3 are off-policy, which is what allows replay buffers. On "
      "CliffWalking, off-policy Q-learning learns the risky edge path that is optimal only once "
      "exploration stops; on-policy SARSA learns the safer path it can actually follow.",
      related=["on-policy", "replay-buffer"], algorithms=["q_learning", "dqn"], envs=["cliff_walking"]),
    T("credit-assignment", "Credit assignment", "Learning from experience",
      "Working out which earlier actions deserve credit for a reward that arrives later.",
      "The longer the delay, the harder it is. Discounting, bootstrapping and eligibility "
      "techniques like GAE are all ways of spreading credit back in time.",
      related=["sparse-reward", "gae", "discount"], envs=["probe3", "mountain_car"]),
    T("sparse-reward", "Sparse reward", "Learning from experience",
      "Almost every step pays nothing, so the agent gets little signal until it stumbles on success.",
      "MountainCar pays only at the flag. Random exploration rarely gets there, so curves stay flat "
      "for a long time and then jump.",
      related=["credit-assignment", "exploration"], envs=["mountain_car", "dyna_maze"]),

    # ----------------------------------------------------- Deep value methods
    T("function-approximation", "Function approximation", "Deep value methods",
      "Representing values or policies with a parameterised function, such as a neural network, instead of a table.",
      "Tables cannot handle continuous or huge state spaces and learn nothing about unseen "
      "states. A network generalises between similar states, at the cost of stability: targets "
      "and estimates now share parameters.",
      related=["one-hot", "replay-buffer", "target-network"], algorithms=["dqn"], envs=["cartpole"]),
    T("replay-buffer", "Replay buffer", "Deep value methods",
      "A memory of past transitions that an off-policy learner samples from for each update.",
      "Sampling at random breaks the correlation between consecutive steps and reuses each "
      "experience many times. The buffer_size metric shows it filling.",
      related=["off-policy", "prioritized-replay"], algorithms=["dqn", "sac"]),
    T("prioritized-replay", "Prioritised replay", "Deep value methods",
      "Replay transitions in proportion to how surprising they were (their TD error).",
      "Rare, informative transitions get revisited more often. Importance-sampling weights "
      "correct the bias this introduces.",
      formula=r"$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}, \quad p_i = |\delta_i| + \epsilon$",
      related=["replay-buffer", "td-error"], algorithms=["dqn_per"]),
    T("target-network", "Target network", "Deep value methods",
      "A slowly updated copy of the network used to compute training targets.",
      "Without it the target moves with every gradient step, and the network chases its own tail. "
      "DQN copies the weights every few hundred updates; SAC and TD3 blend them in slowly.",
      formula=r"$\theta^- \leftarrow \tau\,\theta + (1-\tau)\,\theta^-$",
      related=["bootstrapping", "function-approximation"], algorithms=["dqn", "sac"]),
    T("overestimation", "Overestimation bias", "Deep value methods",
      "Taking the maximum of noisy estimates systematically picks lucky ones, so learned values drift upward.",
      "Double Q-learning and Double DQN separate choosing the best action from evaluating it; TD3 "
      "and SAC take the minimum of two critics.",
      formula=r"$y = r + \gamma\, Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_\theta(s', a')\right)$",
      related=["bootstrapping", "q-function"], algorithms=["double_q", "dqn_double", "td3"]),
    T("dueling", "Dueling architecture", "Deep value methods",
      "A Q-network split into a state-value stream and an advantage stream, recombined into Q.",
      "Many states have values that barely depend on the action; learning V separately makes "
      "those states cheaper to learn.",
      formula=r"$Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$",
      related=["advantage", "q-function"], algorithms=["dueling_dqn"]),
    T("distributional", "Distributional RL", "Deep value methods",
      "Learn the whole distribution of returns rather than only its mean.",
      "Two actions with the same average can carry very different risks. C51 represents the "
      "distribution with 51 bins; acting still uses the mean.",
      formula=r"$Q(s,a) = \mathbb{E}\left[Z(s,a)\right]$",
      related=["q-function", "return"], algorithms=["c51"]),

    # --------------------------------------------------------- Policy methods
    T("policy-gradient", "Policy gradient", "Policy methods",
      "Improve the policy directly by following the gradient of expected return with respect to its parameters.",
      "Actions that led to high return become more likely. Works naturally with stochastic and "
      "continuous policies, but raw estimates are noisy.",
      formula=r"$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s)\; Q^\pi(s,a)\right]$",
      related=["reinforce", "baseline", "actor-critic"], algorithms=["reinforce", "ppo"]),
    T("reinforce", "REINFORCE (likelihood ratio)", "Policy methods",
      "The simplest policy gradient: scale the log-probability gradient of each action by the return that followed.",
      "Unbiased but high variance, which is why its learning curves look noisy and why baselines "
      "were invented.",
      formula=r"$\nabla_\theta J \approx \sum_t G_t \,\nabla_\theta \log \pi_\theta(a_t \mid s_t)$",
      related=["policy-gradient", "baseline", "monte-carlo"], algorithms=["reinforce"]),
    T("baseline", "Baseline", "Policy methods",
      "A value subtracted from returns in the policy gradient to reduce variance without adding bias.",
      "The usual choice is the state value V(s); what remains is the advantage. Compare REINFORCE "
      "with and without a baseline on CartPole over several seeds to see the variance drop.",
      related=["advantage", "variance"], algorithms=["reinforce_baseline"]),
    T("advantage", "Advantage", "Policy methods",
      "How much better an action is than the policy's average in that state.",
      "Positive advantage makes an action more likely, negative less. Using advantages instead of "
      "raw returns is the main variance reduction in actor-critic methods.",
      formula=r"$A(s,a) = Q(s,a) - V(s)$",
      related=["baseline", "gae", "actor-critic"], algorithms=["a2c", "ppo"]),
    T("actor-critic", "Actor-critic", "Policy methods",
      "A policy (the actor) trained with the help of a learned value function (the critic) that judges its actions.",
      "The critic supplies low-variance advantage estimates by bootstrapping; the actor uses them "
      "in a policy-gradient update. A2C, PPO, SAC and TD3 are all actor-critic methods.",
      related=["advantage", "policy-gradient", "value-function"], algorithms=["a2c", "ppo", "sac"]),
    T("gae", "Generalised advantage estimation (GAE)", "Policy methods",
      "An advantage estimate that blends n-step estimates with weights decaying by γλ.",
      "λ trades bias for variance: 0 gives the one-step TD advantage (low variance, biased), 1 gives "
      "Monte Carlo returns minus a baseline (unbiased, noisy). 0.95 is the usual compromise.",
      formula=r"$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \,\delta_{t+l}$",
      related=["advantage", "td-error", "credit-assignment"], algorithms=["ppo"]),
    T("entropy", "Entropy bonus", "Policy methods",
      "A term rewarding the policy for staying random, which keeps it exploring.",
      "Policy entropy usually falls as training converges. Too large a bonus and the policy never "
      "commits; zero and it can collapse onto a poor action early.",
      formula=r"$\mathcal{H}\left(\pi(\cdot \mid s)\right) = -\sum_a \pi(a \mid s)\log \pi(a \mid s)$",
      related=["exploration", "max-entropy"], algorithms=["ppo", "a2c"]),
    T("ppo-clip", "PPO clipping (trust region)", "Policy methods",
      "Limit how far one update can change the policy by clipping the probability ratio.",
      "Large policy steps can wreck a policy that took hours to learn. PPO stops the objective from "
      "rewarding changes beyond 1 ± ε, approximating TRPO's trust region with a simple loss.",
      formula=r"$L = \mathbb{E}_t\left[\min\left(r_t \hat{A}_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\right)\right],\quad r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$",
      related=["kl", "policy-gradient"], algorithms=["ppo", "ppo_continuous"]),
    T("kl", "KL divergence", "Policy methods",
      "A measure of how different two probability distributions are; used to track how far a policy moved.",
      "PPO reports approx_kl per update. Values far above about 0.02 suggest updates are too "
      "aggressive (learning rate or epochs too high).",
      formula=r"$D_{\mathrm{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$",
      related=["ppo-clip"], algorithms=["ppo"]),

    # ----------------------------------------------------- Continuous control
    T("gaussian-policy", "Gaussian policy", "Continuous control",
      "A stochastic policy for continuous actions: a network outputs a mean, and actions are sampled around it.",
      "The spread (standard deviation) controls exploration and is usually learned. Gaussian PPO "
      "and SAC use one; SAC squashes samples through tanh to respect action bounds.",
      formula=r"$\pi_\theta(a \mid s) = \mathcal{N}\!\left(\mu_\theta(s),\ \sigma^2\right)$",
      related=["squashing", "max-entropy"], algorithms=["ppo_continuous", "sac"], envs=["pendulum"]),
    T("dpg", "Deterministic policy gradient", "Continuous control",
      "Improve a deterministic continuous policy by following the critic's gradient with respect to the action.",
      "The actor learns to output actions the critic rates higher. DDPG and TD3 use it, adding "
      "noise to actions only for exploration.",
      formula=r"$\nabla_\theta J = \mathbb{E}\left[\nabla_a Q(s,a)\big|_{a=\mu_\theta(s)}\,\nabla_\theta \mu_\theta(s)\right]$",
      related=["actor-critic", "overestimation"], algorithms=["ddpg", "td3"]),
    T("max-entropy", "Maximum-entropy RL", "Continuous control",
      "Maximise reward plus the policy's entropy, weighted by a temperature α.",
      "The agent is rewarded for achieving the task in as many ways as possible, which makes "
      "exploration robust. SAC tunes α automatically; its alpha metric shows the current value.",
      formula=r"$J(\pi) = \sum_t \mathbb{E}\left[r_t + \alpha\,\mathcal{H}\left(\pi(\cdot \mid s_t)\right)\right]$",
      related=["entropy", "gaussian-policy"], algorithms=["sac"]),
    T("squashing", "Action squashing", "Continuous control",
      "Passing sampled actions through tanh so they always lie within the environment's bounds.",
      "Squashing changes the action's probability, so its log-probability needs a correction term; "
      "forgetting it is a classic SAC bug.",
      formula=r"$a = \tanh(u),\quad \log\pi(a \mid s) = \log\mu(u \mid s) - \sum_i \log\left(1 - \tanh^2(u_i)\right)$",
      related=["gaussian-policy", "max-entropy"], algorithms=["sac"]),
    T("obs-normalization", "Observation normalisation", "Continuous control",
      "Rescaling each observation dimension by a running mean and standard deviation.",
      "MuJoCo observations mix positions near 1 with velocities near 10; without normalisation "
      "the network sees mostly the large numbers.",
      formula=r"$\tilde{s} = \frac{s - \mu}{\sigma}$",
      related=["function-approximation"], algorithms=["sac", "ppo_continuous"], envs=["half_cheetah"]),

    # ---------------------------------------------------------------- Planning
    T("model", "Model (of the environment)", "Planning",
      "A learned or given predictor of the next state and reward for a state and action.",
      "With a model the agent can imagine experience instead of collecting it. Model errors are the "
      "risk: the agent may exploit mistakes in its own model.",
      related=["planning", "dyna"], algorithms=["dyna_q"]),
    T("planning", "Planning", "Planning",
      "Improving a policy or values using a model, without acting in the real environment.",
      "In Dyna-Q, planning means replaying remembered transitions and applying ordinary "
      "Q-learning updates to them.",
      related=["model", "dyna"], algorithms=["dyna_q"], envs=["dyna_maze"]),
    T("dyna", "Dyna", "Planning",
      "An architecture that interleaves learning from real experience, learning a model, and planning with it.",
      "Each real step is followed by n imagined updates. On the Dyna maze, 50 planning steps solve "
      "the task in a handful of real episodes.",
      related=["model", "planning", "sample-efficiency"], algorithms=["dyna_q"], envs=["dyna_maze"]),

    # ------------------------------------------------------- Doing experiments
    T("seed", "Seed", "Doing experiments",
      "The number that fixes every random draw in a run, making it reproducible.",
      "Two seeds of the same configuration can produce very different curves. Results should be "
      "reported across several seeds; the testbed pools them into a mean with a ±1σ band.",
      related=["variance", "confidence-band"]),
    T("variance", "Variance across seeds", "Doing experiments",
      "How much results change between runs that differ only in their random seed.",
      "RL is notoriously high-variance: a single lucky run can make a poor method look good. If "
      "two methods' bands overlap heavily, you cannot yet say one is better.",
      formula=r"$\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2}$",
      related=["seed", "confidence-band"]),
    T("confidence-band", "±1σ band", "Doing experiments",
      "The shaded region around a mean curve, one standard deviation either side, across seeds.",
      "It shows the spread of individual runs, not the uncertainty of the mean. The mean's own "
      "uncertainty, the standard error, shrinks with more seeds; the band does not.",
      formula=r"$\text{standard error} = \frac{\sigma}{\sqrt{n}}$",
      related=["variance", "seed"]),
    T("learning-curve", "Learning curve", "Doing experiments",
      "A plot of performance (usually return) against training progress (episodes or steps).",
      "Read its shape, not just its end: how quickly it rises, how noisy it is, whether it "
      "collapses. Smoothing helps the eye but hides individual episodes.",
      related=["return", "evaluation", "sample-efficiency"]),
    T("evaluation", "Greedy evaluation", "Doing experiments",
      "Measuring a policy with exploration switched off, separate from training episodes.",
      "Training returns include the cost of exploring; evaluations show what has been learned. "
      "The testbed runs one every eval_freq episodes and plots them as a dashed line.",
      related=["greedy", "learning-curve"]),
    T("sample-efficiency", "Sample efficiency", "Doing experiments",
      "How much experience (environment steps) a method needs to reach a given performance.",
      "Off-policy and model-based methods tend to be more sample-efficient; on-policy methods "
      "often need more steps but less tuning. Compare curves on the same x-axis.",
      related=["off-policy", "dyna"]),
    T("threshold", "Solved threshold", "Doing experiments",
      "The return at which a task counts as solved, as defined by the environment's authors.",
      "CartPole is solved at a mean return of 475. Charts draw it as a dashed line; comparisons "
      "report how many episodes each configuration needed to reach it.",
      related=["learning-curve"], envs=["cartpole"]),
    T("hyperparameter", "Hyperparameter", "Doing experiments",
      "A setting chosen before training, such as the learning rate, discount or network size.",
      "RL methods are often more sensitive to hyperparameters than to the choice of method. "
      "Change one at a time, and compare across seeds.",
      related=["sweep", "step-size"]),
    T("sweep", "Hyperparameter sweep", "Doing experiments",
      "Running many configurations systematically to find good settings and learn how sensitive a method is.",
      "Grid search tries every combination; random search samples them and usually finds good "
      "settings with fewer runs. The sweep page ranks configurations and shows each parameter's "
      "effect.",
      related=["hyperparameter", "seed"]),
    T("probe", "Diagnostic probe", "Doing experiments",
      "A tiny environment with a known answer that any correct implementation solves in seconds.",
      "If an algorithm fails a probe, the problem is a bug or a setting, not task difficulty. "
      "Run them before trusting a new implementation.",
      related=["credit-assignment"], envs=["probe1", "probe2", "probe3"]),
    T("checkpoint", "Checkpoint", "Doing experiments",
      "A saved copy of an agent's parameters during or after training.",
      "Runs save the best-evaluating checkpoint; Keep model copies it to Models so it can be "
      "re-evaluated or watched in Visualize after the run is deleted.",
      related=["evaluation"]),
]

BY_ID = {g["id"]: g for g in GLOSSARY}
