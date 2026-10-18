"""Mathematical theory for each algorithm, rendered with KaTeX on its page.

Markdown with $inline$ LaTeX. Moved here from the frontend so algorithm
documentation has exactly one home: the Python package. `build_spec` reads
this table, so nothing else needs to reference it.
"""

THEORY = {
    'pets': (
        '**Origin:** Chua, Calandra, McAllister & Levine, NeurIPS 2018.\n\n'
        '**Model:** an ensemble $\\{f_{\\theta_e}\\}_{e=1}^{E}$ trained on bootstrapped data to predict '
        '$\\Delta s = s_{t+1} - s_t$, the reward and termination from $(s_t, a_t)$. Disagreement between '
        'members approximates epistemic uncertainty.\n\n'
        '**Planning (MPC):** at every step solve '
        '$\\max_{a_{t:t+H}} \\mathbb{E}\\left[\\sum_{k=0}^{H-1} \\gamma^k \\hat r_{t+k}\\right]$ '
        'approximately with the cross-entropy method, execute $a_t$ only, and replan.\n\n'
        '**CEM:** sample $N$ sequences from $\\mathcal{N}(\\mu, \\sigma^2)$, keep the top $K$ by imagined '
        'return, set $\\mu, \\sigma$ to their mean and spread, repeat.\n'
    ),
    'dreamer_lite': (
        '**Origin:** Hafner et al., *Dream to Control* (2020), *DreamerV3* (2023); here without the latent '
        'image model.\n\n'
        '**λ-returns in imagination:** $G_t^\\lambda = \\hat r_t + \\gamma \\hat c_t\\left[(1-\\lambda) V(\\hat s_{t+1}) '
        '+ \\lambda G_{t+1}^\\lambda\\right]$, with $\\hat c_t$ the predicted probability of continuing.\n\n'
        '**Actor:** continuous actions maximise $\\mathbb{E}[G^\\lambda]$ by backpropagating through the '
        'differentiable model (dynamics backpropagation); discrete actions use '
        '$\\nabla \\log \\pi(a\\mid s)\\,(G^\\lambda - V(s))$.\n\n'
        '**Critic:** $\\min_\\psi \\mathbb{E}\\left[(V_\\psi(\\hat s_t) - G_t^\\lambda)^2\\right]$ on imagined states.\n'
    ),
    'ppo_continuous': (
        '**Origin:** Schulman et al. 2017, with a Gaussian policy as in their MuJoCo experiments.\n\n'
        '**Policy:** $\\pi_\\theta(a|s)=\\mathcal{N}(\\mu_\\theta(s),\\,\\mathrm{diag}(\\sigma^2))$ '
        'with $\\log\\sigma$ a free parameter, and $\\log\\pi(a|s)=\\sum_i \\log \\mathcal{N}(a_i;\\mu_i,\\sigma_i)$.\n\n'
        '**Objective:** $L^{CLIP}=\\mathbb{E}_t[\\min(r_t\\hat A_t,\\ \\mathrm{clip}(r_t,1-\\epsilon,1+\\epsilon)\\hat A_t)]$ '
        'with $r_t=\\pi_\\theta(a_t|s_t)/\\pi_{old}(a_t|s_t)$ and $\\hat A_t$ from GAE($\\lambda$).\n\n'
        '**Bounds:** sampling happens in $[-1,1]$; the environment receives '
        '$a_{env}=l+(\\mathrm{clip}(a,-1,1)+1)(h-l)/2$, and the unclipped $a$ is kept for $\\log\\pi$.\n'
    ),
    'random': (
        '**Origin:** Baseline (Sutton & Barto, 2018). No learning — uniform $\\pi(a|s)=1/|A|$.\n'
        '\n'
        '**Formula:** $a_t \\sim \\mathcal{U}(A)$\n'
        '\n'
        '**Where it comes from:** Sanity check; any learner must beat $\\mathbb{E}[R]$ of random.'
    ),
    'q_learning': (
        '**Origin:** Watkins 1989, tabular Q-learning.\n'
        '\n'
        "**Bellman optimality:** $Q^*(s,a)=\\mathbb{E}[r+\\gamma \\max_{a'}Q^*(s',a')\\mid s,a]$\n"
        '\n'
        "**Update:** $Q(s,a) \\leftarrow Q(s,a)+\\alpha\\,[r+\\gamma \\max_{a'}Q(s',a')-Q(s,a)]$ \n"
        '\n'
        '**Policy:** $\\epsilon$-greedy $\\pi(a|s)=\\begin{cases}1-\\epsilon+\\epsilon/|A| & a=\\arg\\max Q \\\\ \\epsilon/|A| & \\text{else}\\end{cases}$\n'
        '\n'
        '**Provenance:** Off-policy, bootstraps from max — may overestimate.'
    ),
    'sarsa': (
        '**Origin:** Rummery & Niranjan 1994, on-policy TD.\n'
        '\n'
        "**Update:** $Q(s,a) \\leftarrow Q(s,a)+\\alpha\\,[r+\\gamma Q(s',a')-Q(s,a)]$ where $a'\\sim\\pi(\\cdot|s')$\n"
        '\n'
        "**vs Q-learning:** SARSA is $\\mathbb{E}_{a'\\sim\\pi}[Q]$ expectation over *actual* next action, so safer near cliffs (learns cautious path)."
    ),
    'dqn': (
        '**Origin:** Mnih et al. Nature 2015.\n'
        '\n'
        "**Loss (Huber):** $L(\\theta)=\\mathbb{E}_{(s,a,r,s')\\sim\\mathcal{D}}[\\ell( y - Q(s,a;\\theta))]$ where $y=r+\\gamma \\max_{a'} Q(s',a';\\theta^-)$\n"
        '\n'
        '**Target:** $\\theta^-$ frozen, sync every $C$ steps.\n'
        '\n'
        '**Replay:** $\\mathcal{D}$ uniform $50k$, $\\epsilon$-greedy annealed $1.0\\to0.02$ over $10k$ steps.\n'
        '\n'
        '**Architecture:** $obs \\to 128 \\to ReLU \\to 128 \\to ReLU \\to |A|$'
    ),
    'dqn_double': (
        '**Origin:** Van Hasselt et al. 2016 — fixes overestimation.\n'
        '\n'
        "**Double estimator:** $y = r + \\gamma Q(s', \\arg\\max_{a'} Q(s',a';\\theta); \\theta^-)$\n"
        '\n'
        'Selection with online $\\theta$, evaluation with target $\\theta^-$ decouples max. Else identical to DQN.'
    ),
    'reinforce': (
        '**Origin:** Williams 1992 REINFORCE.\n'
        '\n'
        '**Objective:** $J(\\theta)=\\mathbb{E}_{\\tau\\sim\\pi_\\theta}[G_0]$, $G_t=\\sum_k \\gamma^k r_{t+k}$\n'
        '\n'
        '**Gradient:** $\\nabla J = \\mathbb{E}[\\sum_t \\nabla \\log \\pi_\\theta(a_t|s_t)\\, G_t ]$\n'
        '\n'
        '**Entropy bonus:** $+c_H H(\\pi)$ for exploration.\n'
        '\n'
        '**Baseline variant:** subtract $b(s)=V(s)$ → $G_t - b(s_t)$ reduces variance.'
    ),
    'reinforce_baseline': (
        '**Origin:** Williams + baseline (Sutton). Same as REINFORCE but learns $V(s)$ with $L_V=\\|G_t-V(s_t)\\|^2$ via separate Adam ($\\alpha_V$).\n'
        '\n'
        '**Advantage:** $A_t=G_t-V(s_t)$ normalized per episode.'
    ),
    'a2c': (
        '**Origin:** Mnih et al. 2016 A2C (synchronous A3C).\n'
        '\n'
        "**Advantage:** $A(s,a)=G^{(n)}-V(s)$ or $r+\\gamma V(s')-V(s)$\n"
        '\n'
        '**Loss:** $L = -\\mathbb{E}[\\log\\pi(a|s)A] + c_v \\|G-V\\|^2 - c_e H(\\pi)$\n'
        '\n'
        'Shared trunk $128$→ policy/value heads, $\\gamma=0.99$.'
    ),
    'ppo': (
        '**Origin:** Schulman et al. 2017.\n'
        '\n'
        '**Clipped surrogate:** $L^{CLIP}=\\mathbb{E}[\\min(r_t(\\theta)A_t, {\\rm clip}(r_t,1-\\epsilon,1+\\epsilon)A_t)]$ with $r_t=\\pi_\\theta/\\pi_{old}$\n'
        '\n'
        "**GAE:** $\\hat A_t = \\sum_l (\\gamma\\lambda)^l \\delta_{t+l}$, $\\delta=r+\\gamma V'-V$\n"
        '\n'
        '$K=10$ epochs, minibatch $256$, $\\epsilon=0.2$, Tanh trunk, approx KL logged.'
    ),
}
