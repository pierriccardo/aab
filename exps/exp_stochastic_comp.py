from libmab.envs.stochastic import GaussianEnv
from libmab.learners import UCB, EpsilonGreedy, Greedy
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

E = 10
T = 10**5
arms = [.2, .5, .4, .6, .23, .34]
K = len(arms)
sigma = .5

# ------------------------------
# Parameters
# ------------------------------

env = GaussianEnv(arms, sigma)
opt_arm = arms[env.opt_arm()]

bandits = [
    UCB(K, T, sigma=sigma),
    EpsilonGreedy(K, T, epsilon=lambda x: .05),
]

regrets = np.zeros((len(bandits), E, T))
rewards = np.zeros((len(bandits), E, T))
armpull = np.zeros((len(bandits), E, K))


for e in tqdm(range(E)):
    for t in range(T):
        for b_id, bandit in enumerate(bandits):
            arm = bandit.pull_arm()
            reward = env.reward(arm, t, e)
            bandit.update(reward, arm)

            #  update data for visualization
            rewards[b_id, e, t] = reward
            regrets[b_id, e, t] = opt_arm - arms[arm]
            armpull[b_id, e, arm] += 1

    for b in bandits:
        b.reset()


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
ax.grid(True, ls='--', lw=.5)
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
ax.grid(True, ls='--', lw=.5)
plt.show()