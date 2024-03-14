from libmab.learners import CUCB, CRandom, Fixed
from libmab.attackers import OracleCombinatorialAttacker
from libmab.envs.combinatorial import CombinatorialGaussianEnv
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

E = 3
T = 10**5
arms   = np.array([.11, .24, .35, .33, .12, .21, .15, .14])
target = np.array([ 1,   0,   0,   0,   1,   0,   1,   1])
K = len(arms)
d = 4
sigma = .1
epsilon = .05


env = CombinatorialGaussianEnv(arms, sigma=sigma, d=d)

def oracle(estimates, d):
    indexes = np.argpartition(estimates, -d)[-d:]
    superarm = np.zeros(len(estimates))
    superarm[indexes] = 1
    return superarm

bandits = [
    CRandom(K, T),
    CUCB(K, T, oracle, sigma=sigma, d=d),
    CUCB(K, T, oracle, sigma=sigma, d=d),
]

labels = [
    "Random",
    "CUCB",
    "Attacked CUCB"
]

attackers = [
    None,
    None,
    OracleCombinatorialAttacker(K, T, target, arms, epsilon=epsilon),
]

regrets = np.zeros((len(bandits), E, T))
rewards = np.zeros((len(bandits), E, T))
armpull = np.zeros((len(bandits), E, K))
attacks = np.zeros((len(attackers), E, T))


for e in tqdm(range(E)):
    for t in tqdm(range(T)):
        rewardvec = env.rewardvec(e, t)
        for b_id, (bandit, attacker) in enumerate(zip(bandits, attackers)):
            arm = bandit.pull_arm()
            attack = attacker.attack(reward, arm) if attacker is not None else np.zeros(K)
            reward = rewardvec * arm
            bandit.update(reward - attack, arm)

            #  update data for visualization
            rewards[b_id, e, t] = np.sum(env.pseudo_reward(arm)) #np.sum(reward)
            regrets[b_id, e, t] = np.sum(env.pseudo_reward(env.opt_arm()) - env.pseudo_reward(arm))
            armpull[b_id, e, :] += arm
            attacks[b_id, e, t] += np.sum(attack)

    for b in bandits:
        print(b)
        b.reset()

x = [*range(T)]

# ----- Regrets -----
fig, ax = plt.subplots()
for b_id, bandit in enumerate(bandits):
    label = bandit.__class__.__name__
    y = np.mean(np.cumsum(regrets, axis=2), axis=1)[b_id]
    c = 1.96 * np.std(np.cumsum(regrets, axis=2), axis=1)[b_id] / np.sqrt(E)
    ax.plot(x, y, label=labels[b_id])
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
    ax.plot(x, y, label=labels[b_id])
ax.legend()
ax.set_title("Cumulative Rewards")
ax.set_xlabel("t")
ax.set_ylabel("Reward")
ax.grid(True, ls='--', lw=.5)
fig.savefig(f"imgs/rewards.png",
        bbox_inches='tight',
        pad_inches=0.05,
        orientation='landscape')

# ----- Regrets -----
fig, ax = plt.subplots()
for b_id, bandit in enumerate(bandits):
    label = bandit.__class__.__name__
    y = np.mean(np.cumsum(attacks, axis=2), axis=1)[b_id]
    c = 1.96 * np.std(np.cumsum(attacks, axis=2), axis=1)[b_id] / np.sqrt(E)
    ax.plot(x, y, label=labels[b_id])
    ax.fill_between(x, (y - c), (y + c), alpha=.5)
ax.legend()
ax.set_title("Cumulative Attack Cost")
ax.set_xlabel("t")
ax.set_ylabel("Attack Cost")
ax.grid(True, ls='--', lw=.5)
fig.savefig(f"imgs/attack_cost.png",
        bbox_inches='tight',
        pad_inches=0.05,
        orientation='landscape')