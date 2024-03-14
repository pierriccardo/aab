from libmab.learners import UCB
from libmab.attackers import ACEAttacker, OracleAttacker, UCBJunAttacker
from libmab.visualization import plotci, bar_plot, Colors
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------
# Parameters
# ------------------------------

E = 10
T = 10**5
D = 1.1
arms = [D, 0.0]  # true arm means
n_arms = len(arms)
sigma = .1
var = sigma**2
opt_mean = np.max(arms)


# attack
target = np.argmin(arms)  # target arm
delta = 0.05
epsilon = .15  # oracle attack ε param

alpha = epsilon / (D + epsilon)

tstart = int(1.9 * alpha * T)  # starting attack time

# experiments data for visualization

# arm pulls
ucb_not_arms_pulled = np.zeros((E, n_arms))
ucb_ora_arms_pulled = np.zeros((E, n_arms))
ucb_ace_arms_pulled = np.zeros((E, n_arms))
ucb_jun_arms_pulled = np.zeros((E, n_arms))


# rewards
ucb_not_experiments_rewards = np.zeros((E, T))
ucb_ora_experiments_rewards = np.zeros((E, T))
ucb_ace_experiments_rewards = np.zeros((E, T))
ucb_jun_experiments_rewards = np.zeros((E, T))

# regrets
ucb_not_experiments_regrets = np.zeros((E, T))
ucb_ora_experiments_regrets = np.zeros((E, T))
ucb_ace_experiments_regrets = np.zeros((E, T))
ucb_jun_experiments_regrets = np.zeros((E, T))

# attack cost
ora_experiments_attacks = np.zeros((E, T))
ace_experiments_attacks = np.zeros((E, T))
jun_experiments_attacks = np.zeros((E, T))


for e in tqdm(range(E)):
    # baselines non attacked
    ucb_not = UCB(n_arms, sigma=sigma)
    ucb_ora = UCB(n_arms, sigma=sigma)
    ucb_ace = UCB(n_arms, sigma=sigma)
    ucb_jun = UCB(n_arms, sigma=sigma)

    # attackers
    ora = OracleAttacker(n_arms, target, arms, epsilon)
    ace = ACEAttacker(n_arms, target, var, delta=.5)
    jun = UCBJunAttacker(n_arms, target, var, delta=.5, delta0=.5)

    for t in range(T):
        # arm selection
        ucb_not_arm = ucb_not.pull_arm()
        ucb_ora_arm = ucb_ora.pull_arm()
        ucb_ace_arm = ucb_ace.pull_arm()
        ucb_jun_arm = ucb_jun.pull_arm()

        # reward generation from environment
        ucb_not_reward = np.random.normal(arms[ucb_not_arm], var)
        ucb_ora_reward = np.random.normal(arms[ucb_ora_arm], var)
        ucb_ace_reward = np.random.normal(arms[ucb_ace_arm], var)
        ucb_jun_reward = np.random.normal(arms[ucb_jun_arm], var)

        # corruption
        ora_corruption = 0
        ace_corruption = 0
        jun_corruption = 0

        if t > tstart:
            ora_corruption = ora.attack(ucb_ora_reward, ucb_ora_arm)
            ace_corruption = ace.attack(ucb_ace_reward, ucb_ace_arm)
            jun_corruption = jun.attack(ucb_jun_reward, ucb_jun_arm)

        ucb_ora_reward -= ora_corruption
        ucb_ace_reward -= ace_corruption
        ucb_jun_reward -= jun_corruption

        # update learners
        ucb_not.update(ucb_not_reward, ucb_not_arm)
        ucb_ora.update(ucb_ora_reward, ucb_ora_arm)
        ucb_ace.update(ucb_ace_reward, ucb_ace_arm)
        ucb_jun.update(ucb_jun_reward, ucb_jun_arm)

        # update data for visualization
        # arm pulls
        ucb_not_arms_pulled[e, ucb_not_arm] += 1
        ucb_ora_arms_pulled[e, ucb_ora_arm] += 1
        ucb_ace_arms_pulled[e, ucb_ace_arm] += 1
        ucb_jun_arms_pulled[e, ucb_jun_arm] += 1

        # rewards
        ucb_not_experiments_rewards[e, t] = ucb_not_reward
        ucb_ora_experiments_rewards[e, t] = ucb_ora_reward
        ucb_ace_experiments_rewards[e, t] = ucb_ace_reward
        ucb_jun_experiments_rewards[e, t] = ucb_jun_reward

        # regrets
        ucb_not_experiments_regrets[e, t] = opt_mean - arms[ucb_not_arm]
        ucb_ora_experiments_regrets[e, t] = opt_mean - arms[ucb_ora_arm]
        ucb_ace_experiments_regrets[e, t] = opt_mean - arms[ucb_ace_arm]
        ucb_jun_experiments_regrets[e, t] = opt_mean - arms[ucb_jun_arm]

        # attack cost
        ora_experiments_attacks[e, t] = ora_corruption
        ace_experiments_attacks[e, t] = ace_corruption
        jun_experiments_attacks[e, t] = jun_corruption

regrets_not = np.mean(np.sum(ucb_not_experiments_regrets, axis=1), axis=0)
regrets_ora = np.mean(np.sum(ucb_ora_experiments_regrets, axis=1), axis=0)
regrets_ace = np.mean(np.sum(ucb_ace_experiments_regrets, axis=1), axis=0)
regrets_jun = np.mean(np.sum(ucb_jun_experiments_regrets, axis=1), axis=0)

attcost_ora = np.mean(np.sum(ora_experiments_attacks, axis=1), axis=0)
attcost_ace = np.mean(np.sum(ace_experiments_attacks, axis=1), axis=0)
attcost_jun = np.mean(np.sum(jun_experiments_attacks, axis=1), axis=0)


print(f"""
    time start = {tstart}

    Attack cost:
        Oracle : {attcost_ora}
        ACE    : {attcost_ace}
        JunUCB : {attcost_jun}

    Total regret:
        Oracle : {regrets_ora}
        ACE    : {regrets_ace}
        JunUCB : {regrets_jun}

    Tot regret / Tot cost
        Oracle : {regrets_ora / attcost_ora}
        ACE    : {regrets_ace / attcost_ace}
        JunUCB : {regrets_jun / attcost_jun}

      """)

EXT = "png"
SAVEPATH = "exps_icml24/baselines"
OFFSET = 30
x = [*range(T)]
os.makedirs(SAVEPATH, exist_ok=True)

fig_regrets = plt.figure()
fig_rewards = plt.figure()
fig_attacks = plt.figure()
fig_armpull = plt.figure()

# ----- Regrets -----
ax1 = fig_regrets.add_subplot(111)
plotci(ax1, x, E, ucb_not_experiments_regrets, Colors.red, "UCB normal")
plotci(ax1, x, E, ucb_ora_experiments_regrets, Colors.blue, "UCB attacked Oracle")
plotci(ax1, x, E, ucb_ace_experiments_regrets, Colors.green, "UCB attacked ACE")
plotci(ax1, x, E, ucb_jun_experiments_regrets, Colors.orange, "UCB attacked Jun UCB")
ax1.axvline(tstart, color="r", ls="--", lw=".9")
ax1.text(tstart - OFFSET, np.median(ax1.get_yticks()), f't\'={tstart}', color="r", ha='right', va='center', fontsize=9, rotation='vertical')
ax1.legend()
ax1.set_title("Cumulative Regret")
ax1.set_xlabel("t")
ax1.set_ylabel("regret")
ax1.grid(True, ls='--', lw=.5)
fig_regrets.savefig(
    f"{SAVEPATH}/baselines_regrets.{EXT}",
    bbox_inches='tight',
    pad_inches=0.05,
    orientation='landscape'
)

# ----- Rewards -----
ax2 = fig_rewards.add_subplot(111)
ax2.plot(x, np.mean(np.cumsum(ucb_not_experiments_rewards, axis=1), axis=0), color=Colors.red, label="UCB normal")
ax2.plot(x, np.mean(np.cumsum(ucb_ora_experiments_rewards, axis=1), axis=0), color=Colors.blue, label="UCB attacked Oracle")
ax2.plot(x, np.mean(np.cumsum(ucb_ace_experiments_rewards, axis=1), axis=0), color=Colors.green, label="UCB attacked ACE")
ax2.plot(x, np.mean(np.cumsum(ucb_jun_experiments_rewards, axis=1), axis=0), color=Colors.orange, label="UCB attacked Jun UCB")
ax2.axvline(tstart, color="r", ls="--", lw=".9", label="t'")
ax2.legend()
ax2.set_title("Cumulative Rewards")
ax2.set_xlabel("t")
ax2.set_ylabel("Reward")
ax2.grid(True, ls='--', lw=.5)
fig_rewards.savefig(
    f"{SAVEPATH}/baselines_rewards.{EXT}",
    bbox_inches='tight',
    pad_inches=0.05,
    orientation='landscape'
)

# ----- Attack cost -----
ax3 = fig_attacks.add_subplot(111)
plotci(ax3, x, E, ora_experiments_attacks, Colors.blue, "Oracle attack cost")
plotci(ax3, x, E, ace_experiments_attacks, Colors.green, "ACE attack cost")
plotci(ax3, x, E, jun_experiments_attacks, Colors.orange, "Jun UCB attack cost")
ax3.axvline(tstart, color="r", ls="--", lw=".9", label="t'")
ax3.legend()
ax3.set_title("Cumulative Attack Cost")
ax3.set_xlabel("t")
ax3.set_ylabel("Attack cost")
ax3.set_xscale("log")
ax3.grid(True, ls='--', lw=.5)
fig_attacks.savefig(
    f"{SAVEPATH}/baselines_attacks.{EXT}",
    bbox_inches='tight',
    pad_inches=0.05,
    orientation='landscape'
)

# ----- Arm pulls -----
ax4 = fig_armpull.add_subplot()
data = {
    "UCB normal": np.mean(ucb_not_arms_pulled, axis=0),
    "UCB Oracle attacked": np.mean(ucb_ora_arms_pulled, axis=0),
    "UCB ACE attacked": np.mean(ucb_ace_arms_pulled, axis=0),
    "UCB Jun attacked": np.mean(ucb_jun_arms_pulled, axis=0),
}
colors = [
    Colors.red,
    Colors.blue,
    Colors.green,
    Colors.orange
]
bar_plot(ax4, data, colors=colors, total_width=0.8, single_width=0.9)
ax4.set_yscale("log")
ax4.set_title("Arm pulls")
ax4.set_xlabel("Arms")
ax4.set_ylabel("Times Pulled")
ax4.set_xticks([*range(n_arms)])
ax4.set_xticklabels([f"arm {a}" if a != target else f"arm {a} (target)" for a in range(n_arms)])
ax4.grid(True, ls='--', lw=.5)
fig_armpull.savefig(
    f"{SAVEPATH}/baselines_armpull.{EXT}",
    bbox_inches='tight',
    pad_inches=0.05,
    orientation='landscape'
)
plt.show()
