import numpy as np
import matplotlib.pyplot as plt
import mablib
import os
from tqdm import tqdm
from utils import *
from config import *

arms = [1, 0]
n_arms = len(arms)
var = (0.1) ** 2
opt_arm = np.max(arms)

# attack
target = np.argmin(arms)
delta = 0.025


att_experiments_rewards = np.zeros((E, T))
egl_experiments_rewards = np.zeros((E, T))
ucb_experiments_rewards = np.zeros((E, T))

att_experiments_attacks = np.zeros((E, T))

att_experiments_regrets = np.zeros((E, T))
egl_experiments_regrets = np.zeros((E, T))
ucb_experiments_regrets = np.zeros((E, T))

att_arms_pulled = np.zeros((E, n_arms))
egl_arms_pulled = np.zeros((E, n_arms))
ucb_arms_pulled = np.zeros((E, n_arms))

for e in tqdm(range(E)):
    # baselines non attacked
    egl = mablib.EpsilonGreedy(n_arms)
    ucb = mablib.UCB(n_arms)

    # baselines attacked
    att = mablib.EpsilonGreedy(n_arms)

    # attacks
    attacker = mablib.EpsilonGreedyAttacker(n_arms, target, var, delta)

    for t in range(T):
        att_arm = target if t == 0 else att.pull_arm()
        egl_arm = egl.pull_arm()
        ucb_arm = ucb.pull_arm()

        att_arms_pulled[e, att_arm] += 1
        egl_arms_pulled[e, egl_arm] += 1
        ucb_arms_pulled[e, ucb_arm] += 1

        att_reward = np.random.normal(arms[att_arm], var)
        egl_reward = np.random.normal(arms[egl_arm], var)
        ucb_reward = np.random.normal(arms[ucb_arm], var)

        # attack
        alpha = attacker.attack(att_reward, att_arm)
        att_reward = att_reward - alpha

        att.update(att_reward, att_arm)
        egl.update(egl_reward, egl_arm)
        ucb.update(ucb_reward, ucb_arm)

        att_experiments_rewards[e, t] = att_reward
        egl_experiments_rewards[e, t] = egl_reward
        ucb_experiments_rewards[e, t] = ucb_reward

        att_experiments_attacks[e, t] = alpha

        att_experiments_regrets[e, t] = opt_arm - arms[att_arm]
        egl_experiments_regrets[e, t] = opt_arm - arms[egl_arm]
        ucb_experiments_regrets[e, t] = opt_arm - arms[ucb_arm]

x = [*range(T)]

fig_regrets = plt.figure()
fig_rewards = plt.figure()
fig_attacks = plt.figure()
fig_armpull = plt.figure()

os.makedirs("imgs", exist_ok=True)

# ----- Regrets -----
ax1 = fig_regrets.add_subplot(111)
plotci(ax1, x, E, att_experiments_regrets, JUN_EGL_COLOR, JUN_EGL_LABEL)
plotci(ax1, x, E, egl_experiments_regrets, EGL_COLOR, EGL_LABEL)
plotci(ax1, x, E, ucb_experiments_regrets, UCB_COLOR, UCB_LABEL)
ax1.legend()
ax1.set_title("Cumulative Regret")
ax1.set_xlabel("t")
ax1.set_ylabel("regret")
fig_regrets.savefig("imgs/jun_regrets")


# ----- Rewards -----
ax2 = fig_rewards.add_subplot(111)
ax2.plot(
    x,
    np.mean(np.cumsum(att_experiments_rewards, axis=1), axis=0),
    color=JUN_EGL_COLOR,
    label=JUN_EGL_LABEL,
)
ax2.plot(
    x,
    np.mean(np.cumsum(egl_experiments_rewards, axis=1), axis=0),
    color=EGL_COLOR,
    label=EGL_LABEL,
)
ax2.plot(
    x,
    np.mean(np.cumsum(ucb_experiments_rewards, axis=1), axis=0),
    color=UCB_COLOR,
    label=UCB_LABEL,
)
ax2.legend()
ax2.set_title("Cumulative Rewards")
ax2.set_xlabel("t")
ax2.set_ylabel("reward")
fig_rewards.savefig("imgs/jun_rewards")

# ----- Attack cost -----
ax3 = fig_attacks.add_subplot(111)
ax3.plot(
    x,
    np.mean(np.cumsum(att_experiments_attacks, axis=1), axis=0),
    label="attack cost on e-greedy",
    color=JUN_EGL_COLOR,
)
ax3.legend()
ax3.set_title("Cumulative Attack Cost")
ax3.set_xlabel("t")
ax3.set_ylabel("Attack cost")
# ax3.set_xscale('log')
fig_attacks.savefig("imgs/jun_attacks_cost")


# ----- Arms Pulled -----
ax4 = fig_armpull.add_subplot()
data = {
    "attacked e-greedy": np.mean(att_arms_pulled, axis=0),
    "e-greedy": np.mean(egl_arms_pulled, axis=0),
    "UCB": np.mean(ucb_arms_pulled, axis=0),
}
colors = [
    JUN_EGL_COLOR,
    EGL_COLOR,
    UCB_COLOR,
]
bar_plot(ax4, data, colors=colors, total_width=0.8, single_width=0.9)
ax4.set_yscale("log")
ax4.set_title("Arm pulls")
ax4.set_xlabel("Arms")
ax4.set_ylabel("Times Pulled")
ax4.set_xticks([*range(n_arms)])
ax4.set_xticklabels(
    [f"arm {a}" if a != target else f"arm {a} (target)" for a in range(n_arms)]
)
fig_armpull.savefig("imgs/jun_arm_pulls")
plt.show()
