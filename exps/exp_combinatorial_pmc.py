from libmab.learners import CUCB, CRandom, Fixed
from libmab.envs import CombinatorialGaussianEnv, PMCEnv
from libmab.visualization import Colors
from libmab.utils import save
from tqdm import tqdm
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import copy
import os

# ------------------------------
# Parameters
# ------------------------------
np.random.seed(123)

bgraph = {
    'C': {4: .1, 5: .3},
    'B': {3: .2, 5: .1},
    'A': {5: .4, 3: .7},
    'D': {1: .12, 2: .6, 3: .18, 5: .12},
    'E': {1: .12, 2: .6, 3: .18, 5: .12, 8: .3},
}

E = 2
T = 10**4
#arms = bgraph.keys()
arms = [.2, .5, .6, .9, .3, .21, .35, .34]
K = len(arms)
d = K - K // 2
sigma = 1.1

env = PMCEnv(bgraph, d=d)
env = CombinatorialGaussianEnv(arms, sigma=sigma, d=d)
print(env)

bandits = [
    CUCB(K, T, d),
    CRandom(K, T, d),
]

regrets = np.zeros((len(bandits), E, T))
rewards = np.zeros((len(bandits), E, T))
armpull = np.zeros((len(bandits), E, K))


for e in tqdm(range(E)):
    for t in tqdm(range(T)):
        rewardvec = env.rewardvec(t)
        for b_id, bandit in enumerate(bandits):
            arm = bandit.pull_arm()
            #reward = env.reward(arm)
            reward = rewardvec * arm
            bandit.update(reward, arm)

            #  update data for visualization
            rewards[b_id, e, t] = np.sum(reward)
            regrets[b_id, e, t] = np.sum(env.pseudo_reward(env.opt_arm()) - env.pseudo_reward(arm))
            armpull[b_id, e, :] += arm

    for b in bandits:
        print(b)
        b.reset()

save(
    ['regrets', 'rewards'],
    [regrets, rewards],
    ['UCB', 'lol']
)

x = [*range(T)]

# ----- Regrets -----
fig, ax = plt.subplots()
for b_id, bandit in enumerate(bandits):
    label = bandit.__class__.__name__
    y = np.mean(np.cumsum(regrets, axis=2), axis=1)[b_id]
    c = 1.96 * np.std(np.cumsum(regrets, axis=2), axis=1)[b_id] / np.sqrt(E)
    ax.plot(x, y, label=label)
    ax.fill_between(x, (y - c), (y + c), alpha=.5)
ax.legend()
ax.set_title("Cumulative Regret")
ax.set_xlabel("t")
ax.set_ylabel("regret")
ax.grid(True, ls='--', lw=.5)
fig.savefig(f"imgs/regret.png",
        bbox_inches='tight',
        pad_inches=0.05,
        orientation='landscape')


# ----- Rewards -----
fig, ax = plt.subplots()
for b_id, bandit in enumerate(bandits):
    label = bandit.__class__.__name__
    y = np.mean(np.cumsum(rewards, axis=2), axis=1)[b_id]
    ax.plot(x, y, label=label)
ax.legend()
ax.set_title("Cumulative Rewards")
ax.set_xlabel("t")
ax.set_ylabel("Reward")
ax.grid(True, ls='--', lw=.5)
fig.savefig(f"imgs/rewards.png",
        bbox_inches='tight',
        pad_inches=0.05,
        orientation='landscape')
