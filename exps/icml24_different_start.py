from libmab.learners import UCB
from libmab.attackers import OracleAttacker
from libmab.visualization import Colors
from matplotlib import pyplot as plt
from tqdm import tqdm

import tikzplotlib as tkz
import matplotlib as mpl
import numpy as np
import os

# ------------------------------
# Parameters
# ------------------------------
seed = 1234
np.random.seed(seed=seed)

E = 15
T = 10**3
TSTARTS = T
D = .5

TS = [*range(0, T, T//TSTARTS)]

arms = [D, 0.0]  # true arm means
n_arms = len(arms)
sigma = .1
var = sigma**2
opt_mean = np.max(arms)


# attack
target = np.argmin(arms)  # target arm
opt = np.argmax(arms)  # target arm
delta = 0.05
epsilon = .05  # oracle attack ε param

# experiments data for visualization
# arm pulls
ucb_arms_pulled = np.zeros((TSTARTS, E, n_arms))
ucb_arms_pulled_attimes = np.zeros((TSTARTS, E, T, n_arms))

# rewards
ucb_experiments_rewards = np.zeros((TSTARTS, E, T))

# regrets
ucb_experiments_regrets = np.zeros((TSTARTS, E, T))

# attack cost
ora_experiments_attacks = np.zeros((TSTARTS, E, T))

for tstart in tqdm(TS):
    tstart = int(tstart)
    if tstart >= TSTARTS:
        continue

    for e in range(E):
        # baselines non attacked
        ucb = UCB(n_arms, sigma=sigma)
        # attackers
        ora = OracleAttacker(n_arms, target, arms, epsilon)

        for t in range(T):
            # arm selection
            ucb_arm = ucb.pull_arm()

            # reward generation from environment
            ucb_reward = np.random.normal(arms[ucb_arm], var)

            # corruption
            ora_corruption = 0

            if t > tstart:
                ora_corruption = ora.attack(ucb_reward, ucb_arm)

            ucb_reward -= ora_corruption

            # update learners
            ucb.update(ucb_reward, ucb_arm)

            # update data for visualization
            ucb_arms_pulled[tstart, e, ucb_arm] += 1  # arm pulls
            ucb_arms_pulled_attimes[tstart, e, t, ucb_arm] += 1
            ucb_experiments_rewards[tstart, e, t] = ucb_reward  # rewards
            ucb_experiments_regrets[tstart, e, t] = opt_mean - arms[ucb_arm]  # regrets
            ora_experiments_attacks[tstart, e, t] = ora_corruption  # attack cost

# ------------------------------
# Visualization
# ------------------------------

EXT = "pdf"
SAVEPATH = "exps_icml24/"
SAVEPATHTEX = "exps_icml24/texs"
EXTRA_TIKZ_PARAM = [
    "every node/.append style={font=\\Large}"
]
os.makedirs(SAVEPATH, exist_ok=True)
os.makedirs(SAVEPATHTEX, exist_ok=True)

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.color'] = 'black'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linewidth'] = .9
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.markerscale'] = .8

RED = 100
x = TS[::RED]


# ----- metrics -----
fig, ax = plt.subplots()

y0 = np.mean(np.sum(ucb_experiments_regrets, axis=2), axis=1)
y1 = np.mean(np.sum(ucb_experiments_rewards, axis=2), axis=1)
y2 = np.mean(np.sum(ora_experiments_attacks, axis=2), axis=1)
c0 = 1.96 * np.mean(np.sum(ucb_experiments_regrets, axis=2), axis=1) / np.sqrt(E)
c1 = 1.96 * np.mean(np.sum(ucb_experiments_rewards, axis=2), axis=1) / np.sqrt(E)
c2 = 1.96 * np.mean(np.sum(ora_experiments_attacks, axis=2), axis=1) / np.sqrt(E)
ax.plot(x, y0[::RED], color=Colors.red,   label=f"Regrets", marker='*', markevery=len(x) // len(ax.get_xticks()))
ax.plot(x, y1[::RED], color=Colors.blue,  label=f"Rewards", marker='v', markevery=len(x) // len(ax.get_xticks()))
ax.plot(x, y2[::RED], color=Colors.green, label=f"Attack Costs", marker='s', markevery=len(x) // len(ax.get_xticks()))
ax.fill_between(x, (y0 - c0)[::RED], (y0 + c0)[::RED], alpha=.1, color=Colors.red)
ax.fill_between(x, (y1 - c1)[::RED], (y1 + c1)[::RED], alpha=.1, color=Colors.blue)
ax.fill_between(x, (y2 - c2)[::RED], (y2 + c2)[::RED], alpha=.1, color=Colors.green)
ax.legend()
ax.set_title("Metrics Comparison")
ax.set_xlabel("Attack Start Time $t^A$")
ax.set_ylabel("Metrics")
fig.tight_layout()
fig.savefig(f"{SAVEPATH}/different_start_metrics.{EXT}",
        bbox_inches='tight',
        pad_inches=0.05,
        orientation='landscape')
tkz.save(f'{SAVEPATHTEX}/different_start_metrics.tex', extra_tikzpicture_parameters=EXTRA_TIKZ_PARAM)

# ----- Arm pulls -----
fig, ax = plt.subplots()
y = np.mean(np.sum(ucb_arms_pulled_attimes, axis=2), axis=1)
c = 1.96 * np.std(np.sum(ucb_arms_pulled_attimes, axis=2), axis=1) / np.sqrt(E)
y0 = y[:, target]
y1 = y[:, opt]
c0 = c[:, target]
c1 = c[:, opt]
ax.plot(x, y0[::RED], color=Colors.red,   label="Target arm", marker='*', markevery=len(x) // len(ax.get_xticks()))
ax.plot(x, y1[::RED], color=Colors.blue,  label="Optimal arm", marker='v', markevery=len(x) // len(ax.get_xticks()))
ax.fill_between(x, (y0 - c0)[::RED], (y0 + c0)[::RED], alpha=.1, color=Colors.red)
ax.fill_between(x, (y1 - c1)[::RED], (y1 + c1)[::RED], alpha=.1, color=Colors.blue)
ax.legend()
ax.set_title("Arm Pulls")
ax.set_xlabel("Attack Start Time $t^A$")
ax.set_ylabel("Times Pulled Target")
fig.tight_layout()
fig.savefig(f"{SAVEPATH}/different_start_armpull.{EXT}",
        bbox_inches='tight',
        pad_inches=0.05,
        orientation='landscape')
tkz.save(f'{SAVEPATHTEX}/different_start_armpull.tex', extra_tikzpicture_parameters=EXTRA_TIKZ_PARAM)
