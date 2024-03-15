from libmab.envs.stochastic import GaussianEnv
from libmab.learners import UCB, EpsilonGreedy
from libmab.attackers import OracleAttacker
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

E = 10
T = 10**5
arms = [0.2, 0.5, 0.4, 0.6, 0.23, 0.34]
K = len(arms)
sigma = 1.1

# ------------------------------
# Parameters
# ------------------------------

env = GaussianEnv(arms, sigma)
opt_arm = arms[env.opt_arm()]
target = np.argmin(arms)

bandits = [UCB(K, T, sigma=sigma), EpsilonGreedy(K, T), UCB(K, T, sigma=sigma)]

attackers = [
    OracleAttacker(K, T, target, arms, epsilon=0.05),
    OracleAttacker(K, T, target, arms, epsilon=0.05),
    None,
]

B = len(bandits)
A = len(attackers)
regrets = np.zeros((B, E, T))
rewards = np.zeros((B, E, T))
armpull = np.zeros((B, E, K))
attacks = np.zeros((A, E, T))


for e in tqdm(range(E)):
    for t in range(T):
        for b_id, (bandit, attacker) in enumerate(zip(bandits, attackers)):
            arm = bandit.pull_arm()
            reward = env.reward(arm, t, e)
            attack = attacker.attack(reward, arm) if attacker is not None else 0
            bandit.update(reward - attack, arm)

            #  update data for visualization
            rewards[b_id, e, t] = reward
            regrets[b_id, e, t] = opt_arm - arms[arm]
            armpull[b_id, e, arm] += 1

    for b, a in zip(bandits, attackers):
        b.reset()
        if a is not None:
            a.reset()


x = [*range(T)]

# ----- Regrets -----
fig, ax = plt.subplots()
for b_id, bandit in enumerate(bandits):
    print(type(bandit))
    label = bandit.__class__.__name__
    y = np.mean(np.cumsum(regrets, axis=2), axis=1)[b_id]
    ax.plot(x, y, label=label)
ax.legend()
ax.set_title("Cumulative Regret")
ax.set_xlabel("t")
ax.set_ylabel("regret")
ax.grid(True, ls="--", lw=0.5)
plt.show()

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
ax.grid(True, ls="--", lw=0.5)
plt.show()
