from libmab.learners import UCB
from libmab.attackers import OracleAttacker
from libmab.visualization import plotci, bar_plot, Colors, getconfint
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import tikzplotlib as tkz
import os

# attacker color C1121F https://coolors.co/palette/780000-c1121f-fdf0d5-003049-669bbc
# learner color 58, 134, 255 3A86FF https://coolors.co/palette/ffbe0b-fb5607-ff006e-8338ec-3a86ff


# ------------------------------
# Parameters
# ------------------------------
seed = 1234
E = 10
T = 10**5
D = 0.5

np.random.seed(seed=seed)

arms = [D, 0.0]  # true arm means
n_arms = len(arms)
sigma = 0.1
var = sigma**2
opt_mean = np.max(arms)


# attack
target = np.argmin(arms)  # target arm
delta = 0.05
epsilon = 0.05  # oracle attack ε param

# alpha paramter
alpha = epsilon / (D + epsilon)

tstart_1 = int(0.5 * alpha * T)  # starting attack time
tstart_2 = int(1.5 * alpha * T)  # starting attack time

# experiments data for visualization

# arm pulls
ucb_time0_arms_pulled = np.zeros((E, n_arms))
ucb_time1_arms_pulled = np.zeros((E, n_arms))
ucb_time2_arms_pulled = np.zeros((E, n_arms))


# rewards
ucb_time0_experiments_rewards = np.zeros((E, T))
ucb_time1_experiments_rewards = np.zeros((E, T))
ucb_time2_experiments_rewards = np.zeros((E, T))


# regrets
ucb_time0_experiments_regrets = np.zeros((E, T))
ucb_time1_experiments_regrets = np.zeros((E, T))
ucb_time2_experiments_regrets = np.zeros((E, T))


# attack cost
ora_time0_experiments_attacks = np.zeros((E, T))
ora_time1_experiments_attacks = np.zeros((E, T))
ora_time2_experiments_attacks = np.zeros((E, T))


for e in tqdm(range(E)):
    # baselines non attacked
    ucb_time0 = UCB(n_arms, sigma=sigma)
    ucb_time1 = UCB(n_arms, sigma=sigma)
    ucb_time2 = UCB(n_arms, sigma=sigma)

    # attackers
    ora_time0 = OracleAttacker(n_arms, target, arms, epsilon)
    ora_time1 = OracleAttacker(n_arms, target, arms, epsilon)
    ora_time2 = OracleAttacker(n_arms, target, arms, epsilon)

    for t in range(T):
        # arm selection
        ucb_time0_arm = ucb_time0.pull_arm()
        ucb_time1_arm = ucb_time1.pull_arm()
        ucb_time2_arm = ucb_time2.pull_arm()

        # reward generation from environment
        ucb_time0_reward = np.random.normal(arms[ucb_time0_arm], var)
        ucb_time1_reward = np.random.normal(arms[ucb_time1_arm], var)
        ucb_time2_reward = np.random.normal(arms[ucb_time2_arm], var)

        # corruption
        ora_time0_corruption = 0
        ora_time1_corruption = 0
        ora_time2_corruption = 0

        ora_time0_corruption = ora_time0.attack(ucb_time0_reward, ucb_time0_arm)
        if t > tstart_1:
            ora_time1_corruption = ora_time1.attack(ucb_time1_reward, ucb_time1_arm)
        if t > tstart_2:
            ora_time2_corruption = ora_time2.attack(ucb_time2_reward, ucb_time2_arm)

        ucb_time0_reward -= ora_time0_corruption
        ucb_time1_reward -= ora_time1_corruption
        ucb_time2_reward -= ora_time2_corruption

        # update learners
        ucb_time0.update(ucb_time0_reward, ucb_time0_arm)
        ucb_time1.update(ucb_time1_reward, ucb_time1_arm)
        ucb_time2.update(ucb_time2_reward, ucb_time2_arm)

        # update data for visualization
        # arm pulls
        ucb_time0_arms_pulled[e, ucb_time0_arm] += 1
        ucb_time1_arms_pulled[e, ucb_time1_arm] += 1
        ucb_time2_arms_pulled[e, ucb_time2_arm] += 1

        # rewards
        ucb_time0_experiments_rewards[e, t] = ucb_time0_reward
        ucb_time1_experiments_rewards[e, t] = ucb_time1_reward
        ucb_time2_experiments_rewards[e, t] = ucb_time2_reward

        # regrets
        ucb_time0_experiments_regrets[e, t] = opt_mean - arms[ucb_time0_arm]
        ucb_time1_experiments_regrets[e, t] = opt_mean - arms[ucb_time1_arm]
        ucb_time2_experiments_regrets[e, t] = opt_mean - arms[ucb_time2_arm]

        # attack cost
        ora_time0_experiments_attacks[e, t] = ora_time0_corruption
        ora_time1_experiments_attacks[e, t] = ora_time1_corruption
        ora_time2_experiments_attacks[e, t] = ora_time2_corruption

attcost_time0 = np.mean(np.sum(ora_time0_experiments_attacks, axis=1), axis=0)
attcost_time1 = np.mean(np.sum(ora_time1_experiments_attacks, axis=1), axis=0)
attcost_time2 = np.mean(np.sum(ora_time2_experiments_attacks, axis=1), axis=0)
regrets_time0 = np.mean(np.sum(ucb_time0_experiments_regrets, axis=1), axis=0)
regrets_time1 = np.mean(np.sum(ucb_time1_experiments_regrets, axis=1), axis=0)
regrets_time2 = np.mean(np.sum(ucb_time2_experiments_regrets, axis=1), axis=0)


print(
    f"""
    time 1 = {tstart_1}
    time 2 = {tstart_2}

    Attack cost:
        time 0 : {attcost_time0}
        time 1 : {attcost_time1}
        time 2 : {attcost_time2}

    Total regret:
        time 0 : {regrets_time0}
        time 1 : {regrets_time1}
        time 2 : {regrets_time2}

    Tot regret / Tot cost
        time 0 : {regrets_time0 / attcost_time0}
        time 1 : {regrets_time1 / attcost_time1}
        time 2 : {regrets_time2 / attcost_time2}

      """
)

# ------------------------------
# Visualization
# ------------------------------

mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["lines.color"] = "black"
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linewidth"] = 0.9
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["legend.fontsize"] = "small"
mpl.rcParams["legend.markerscale"] = 0.8

EXT = "pdf"
SAVEPATH = "exps_icml24/"
SAVEPATHTEX = "exps_icml24/texs"
OFFSET = 30
RED = 1000
LSSTART = "--"
FIGSIZE = (8, 6)
# https://tex.stackexchange.com/questions/278049/tikzpicture-scaling-up-the-axis-ticks-and-label
EXTRA_AXIS_PARAM = [
    "width=9cm",
    "height=7cm",
]
EXTRA_TIKZ_PARAM = ["every node/.append style={font=\\Large}"]
os.makedirs(SAVEPATH, exist_ok=True)
os.makedirs(SAVEPATHTEX, exist_ok=True)

x = [*range(T)][::RED]

# ----- Regrets -----
fig, ax = plt.subplots()
y0 = np.mean(np.cumsum(ucb_time0_experiments_regrets, axis=1), axis=0)
y1 = np.mean(np.cumsum(ucb_time1_experiments_regrets, axis=1), axis=0)
y2 = np.mean(np.cumsum(ucb_time2_experiments_regrets, axis=1), axis=0)
c0 = (
    1.96 * np.std(np.cumsum(ucb_time0_experiments_regrets, axis=1), axis=0) / np.sqrt(E)
)
c1 = (
    1.96 * np.std(np.cumsum(ucb_time1_experiments_regrets, axis=1), axis=0) / np.sqrt(E)
)
c2 = (
    1.96 * np.std(np.cumsum(ucb_time2_experiments_regrets, axis=1), axis=0) / np.sqrt(E)
)
ax.plot(
    x,
    y0[::RED],
    color=Colors.red,
    label=f"UCB OA $t^A$={0}",
    marker="*",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.plot(
    x,
    y1[::RED],
    color=Colors.blue,
    label=f"UCB OA $t^A$={tstart_1}",
    marker="v",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.plot(
    x,
    y2[::RED],
    color=Colors.green,
    label=f"UCB OA $t^A$={tstart_2}",
    marker="s",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.fill_between(x, (y0 - c0)[::RED], (y0 + c0)[::RED], alpha=0.1, color=Colors.red)
ax.fill_between(x, (y1 - c1)[::RED], (y1 + c1)[::RED], alpha=0.1, color=Colors.blue)
ax.fill_between(x, (y2 - c2)[::RED], (y2 + c2)[::RED], alpha=0.1, color=Colors.green)
ax.axvline(tstart_1, color="red", ls=LSSTART, lw=1.2)
# ax.text(tstart_1 - OFFSET, np.median(ax.get_yticks()), f't\'={tstart_1}', ha='righ$t^A$, va='center', rotation='vertical')
ax.axvline(tstart_2, color="red", ls=LSSTART, lw=1.2)
# ax.text(tstart_2 + OFFSET**2, np.median(ax.get_yticks()), f't\'={tstart_2}', ha='lef$t^A$, va='center', rotation='vertical')
ax.legend()
ax.set_title("Cumulative Regret")
ax.set_xlabel("t")
ax.set_ylabel("Regret")
fig.tight_layout()
fig.savefig(
    f"{SAVEPATH}/comparison_regrets.{EXT}",
    bbox_inches="tight",
    pad_inches=0.05,
    orientation="landscape",
)
tkz.save(
    f"{SAVEPATHTEX}/comparison_regrets.tex",
    extra_tikzpicture_parameters=EXTRA_TIKZ_PARAM,
    extra_axis_parameters=EXTRA_AXIS_PARAM,
)

# ----- Rewards -----
fig, ax = plt.subplots()
# ax.plot(x, np.mean(np.cumsum(ucb_time0_experiments_rewards, axis=1), axis=0)[::RED],marker='*', markevery=len(x) // len(ax.get_xticks()), color=Colors.red, label="UCB OA $t^A$=0")
# ax.plot(x, np.mean(np.cumsum(ucb_time1_experiments_rewards, axis=1), axis=0)[::RED],marker='v', markevery=len(x) // len(ax.get_xticks()), color=Colors.blue, label=f"UCB OA $t^A$={tstart_1}")
# ax.plot(x, np.mean(np.cumsum(ucb_time2_experiments_rewards, axis=1), axis=0)[::RED],marker='s', markevery=len(x) // len(ax.get_xticks()), color=Colors.green, label=f"UCB OA $t^A$={tstart_2}")
y0 = np.mean(np.cumsum(ucb_time0_experiments_rewards, axis=1), axis=0)
y1 = np.mean(np.cumsum(ucb_time1_experiments_rewards, axis=1), axis=0)
y2 = np.mean(np.cumsum(ucb_time2_experiments_rewards, axis=1), axis=0)
c0 = (
    1.96 * np.std(np.cumsum(ucb_time0_experiments_rewards, axis=1), axis=0) / np.sqrt(E)
)
c1 = (
    1.96 * np.std(np.cumsum(ucb_time1_experiments_rewards, axis=1), axis=0) / np.sqrt(E)
)
c2 = (
    1.96 * np.std(np.cumsum(ucb_time2_experiments_rewards, axis=1), axis=0) / np.sqrt(E)
)
ax.plot(
    x,
    y0[::RED],
    color=Colors.red,
    label=f"UCB OA $t^A$={0}",
    marker="*",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.plot(
    x,
    y1[::RED],
    color=Colors.blue,
    label=f"UCB OA $t^A$={tstart_1}",
    marker="v",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.plot(
    x,
    y2[::RED],
    color=Colors.green,
    label=f"UCB OA $t^A$={tstart_2}",
    marker="s",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.axvline(tstart_1, color="red", ls=LSSTART, lw=1.2)
# ax.text(tstart_1 - OFFSET, np.median(ax.get_yticks()), f't\'={tstart_1}', ha='righ$t^A$, va='center', rotation='vertical')
ax.axvline(tstart_2, color="red", ls=LSSTART, lw=1.2)
# ax.text(tstart_2 + OFFSET**2, np.median(ax.get_yticks()), f't\'={tstart_2}', ha='lef$t^A$, va='center', rotation='vertical')
ax.legend()
ax.set_title("Cumulative Rewards")
ax.set_xlabel("t")
ax.set_ylabel("Reward")
fig.tight_layout()
fig.savefig(
    f"{SAVEPATH}/comparison_rewards.{EXT}",
    bbox_inches="tight",
    pad_inches=0.05,
    orientation="landscape",
)
tkz.save(
    f"{SAVEPATHTEX}/comparison_rewards.tex",
    extra_tikzpicture_parameters=EXTRA_TIKZ_PARAM,
    extra_axis_parameters=EXTRA_AXIS_PARAM,
)


# ----- Attack cost -----
fig, ax = plt.subplots()
y0 = np.mean(np.cumsum(ora_time0_experiments_attacks, axis=1), axis=0)
y1 = np.mean(np.cumsum(ora_time1_experiments_attacks, axis=1), axis=0)
y2 = np.mean(np.cumsum(ora_time2_experiments_attacks, axis=1), axis=0)
c0 = (
    1.96 * np.std(np.cumsum(ora_time0_experiments_attacks, axis=1), axis=0) / np.sqrt(E)
)
c1 = (
    1.96 * np.std(np.cumsum(ora_time1_experiments_attacks, axis=1), axis=0) / np.sqrt(E)
)
c2 = (
    1.96 * np.std(np.cumsum(ora_time2_experiments_attacks, axis=1), axis=0) / np.sqrt(E)
)
ax.plot(
    x,
    y0[::RED],
    color=Colors.red,
    label=f"OA $t^A$={0}",
    marker="*",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.plot(
    x,
    y1[::RED],
    color=Colors.blue,
    label=f"OA $t^A$={tstart_1}",
    marker="v",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.plot(
    x,
    y2[::RED],
    color=Colors.green,
    label=f"OA $t^A$={tstart_2}",
    marker="s",
    markevery=len(x) // len(ax.get_xticks()),
)
ax.fill_between(x, (y0 - c0)[::RED], (y0 + c0)[::RED], alpha=0.1, color=Colors.red)
ax.fill_between(x, (y1 - c1)[::RED], (y1 + c1)[::RED], alpha=0.1, color=Colors.blue)
ax.fill_between(x, (y2 - c2)[::RED], (y2 + c2)[::RED], alpha=0.1, color=Colors.green)
ax.axvline(tstart_1, color="red", ls=LSSTART, lw=1.2)
# ax.text(tstart_1 - OFFSET, np.median(ax.get_yticks()), f't\'={tstart_1}', ha='righ$t^A$, va='center', rotation='vertical')
ax.axvline(tstart_2, color="red", ls=LSSTART, lw=1.2)
# ax.text(tstart_2 + OFFSET**2, np.median(ax.get_yticks()), f't\'={tstart_2}', ha='lef$t^A$, va='center', rotation='vertical')
ax.legend()
ax.set_title("Cumulative Attack Cost")
ax.set_xlabel("t")
ax.set_ylabel("Attack Cost")
fig.tight_layout()
fig.savefig(
    f"{SAVEPATH}/comparison_attacks.{EXT}",
    bbox_inches="tight",
    pad_inches=0.05,
    orientation="landscape",
)
tkz.save(
    f"{SAVEPATHTEX}/comparison_attacks.tex",
    extra_tikzpicture_parameters=EXTRA_TIKZ_PARAM,
    extra_axis_parameters=EXTRA_AXIS_PARAM,
)

# ----- Arm pulls -----
fig, ax = plt.subplots()
data = {
    f"UCB OA $t^A$={0}": np.mean(ucb_time0_arms_pulled, axis=0),
    f"UCB OA $t^A$={tstart_1}": np.mean(ucb_time1_arms_pulled, axis=0),
    f"UCB OA $t^A$={tstart_2}": np.mean(ucb_time2_arms_pulled, axis=0),
}
colors = [
    Colors.red,
    Colors.blue,
    Colors.green,
]
bar_plot(ax, data, colors=colors, total_width=0.8, single_width=0.9)
ax.ticklabel_format(style="sci", axis="both")
ax.set_yscale("log")
ax.set_title("Arm Pulls")
ax.set_xlabel("Arms")
ax.set_ylabel("Times Pulled")
ax.set_xticks([*range(n_arms)])
ax.set_xticklabels(
    [f"arm {a}" if a != target else f"arm {a} (target)" for a in range(n_arms)]
)
ax.grid(False)
fig.tight_layout()
fig.savefig(
    f"{SAVEPATH}/comparison_armpull.{EXT}",
    bbox_inches="tight",
    pad_inches=0.05,
    orientation="landscape",
)
tkz.save(
    f"{SAVEPATHTEX}/comparison_armpull.tex",
    extra_tikzpicture_parameters=EXTRA_TIKZ_PARAM,
)
