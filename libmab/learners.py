# from libmab.visualization import plotci, bar_plot
from visualization import plotci, bar_plot
from tqdm import tqdm

import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import numpy as np
import abc


class Learner:
    def __init__(self, n_arms: int, T: int) -> None:
        self.n_arms = n_arms
        self.t = 0
        self.arm_pulls = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)
        self.rewards = []

    def update(self, reward, arm) -> None:
        self.rewards.append(reward)

        self.estimates[arm] = (self.estimates[arm] * self.arm_pulls[arm] + reward) / (
            self.arm_pulls[arm] + 1
        )
        self.arm_pulls[arm] += 1
        self.t += 1

    @abc.abstractclassmethod
    def pull_arm(self) -> int:
        pass


class Greedy(Learner):
    def __init__(self, n_arms: int) -> None:
        super().__init__(n_arms)

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        else:
            return np.argmax(self.estimates)


class EpsilonGreedy(Learner):
    def __init__(self, n_arms: int, epsilon: callable = lambda x: 1 / x):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        else:
            if np.random.random() <= self.epsilon(self.t):
                return np.random.choice(range(self.n_arms))
            else:
                return np.argmax(self.estimates)


class UCBClassic(Learner):
    def __init__(self, n_arms: int, c: int = 2):
        super().__init__(n_arms)
        self.c = c

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        else:
            exploration = self.c * np.log(self.t) / self.arm_pulls
            exploration = np.sqrt(exploration)
            sel = np.add(self.estimates, exploration)
            return np.argmax(sel)


class UCB(Learner):
    def __init__(self, n_arms: int, sigma: float = 1) -> None:
        super().__init__(n_arms)
        self.sigma = sigma

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        else:
            #exploration = np.log(10**5) / self.arm_pulls
            exploration = np.log(self.t) / self.arm_pulls
            exploration = 3 * self.sigma * np.sqrt(exploration)
            sel = np.add(self.estimates, exploration)
            return np.argmax(sel)


if __name__ == "__main__":
    T = 10000
    E = 100
    arms = [0.49, 0.67, 0.35, 0.90]
    n_arms = len(arms)
    var = (1) ** 2
    opt_arm = np.max(arms)

    # pulled arms
    grl_arms_pulled = np.zeros((E, n_arms))
    egl_arms_pulled = np.zeros((E, n_arms))
    ucb_arms_pulled = np.zeros((E, n_arms))

    # rewards
    grl_experiments_rewards = np.zeros((E, T))
    egl_experiments_rewards = np.zeros((E, T))
    ucb_experiments_rewards = np.zeros((E, T))

    # regrets
    grl_experiments_regrets = np.zeros((E, T))
    egl_experiments_regrets = np.zeros((E, T))
    ucb_experiments_regrets = np.zeros((E, T))

    for e in tqdm(range(E)):
        # baselines
        grl = Greedy(n_arms)
        egl = EpsilonGreedy(n_arms)
        ucb = UCB(n_arms)

        for t in range(T):
            # arm selection
            grl_arm = grl.pull_arm()
            egl_arm = egl.pull_arm()
            ucb_arm = ucb.pull_arm()

            # reward from env
            grl_reward = np.random.normal(arms[grl_arm], var)
            egl_reward = np.random.normal(arms[egl_arm], var)
            ucb_reward = np.random.normal(arms[ucb_arm], var)

            # update learner
            grl.update(grl_reward, grl_arm)
            egl.update(egl_reward, egl_arm)
            ucb.update(ucb_reward, ucb_arm)

            # update
            grl_arms_pulled[e, grl_arm] += 1
            egl_arms_pulled[e, egl_arm] += 1
            ucb_arms_pulled[e, ucb_arm] += 1

            grl_experiments_rewards[e, t] = grl_reward
            egl_experiments_rewards[e, t] = egl_reward
            ucb_experiments_rewards[e, t] = ucb_reward

            grl_experiments_regrets[e, t] = opt_arm - arms[grl_arm]
            egl_experiments_regrets[e, t] = opt_arm - arms[egl_arm]
            ucb_experiments_regrets[e, t] = opt_arm - arms[ucb_arm]

    x = [*range(T)]

    GRL_COLOR = "blue"
    EGL_COLOR = "green"
    UCB_COLOR = "red"

    GRL_LABEL = "greedy"
    UCB_LABEL = "UCB"
    EGL_LABEL = "ε-greedy"

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2, width_ratios=[6, 4], height_ratios=[5, 3])
    gs.update(wspace=0.5, hspace=0.5)

    pl.figure()

    # ----- Regrets -----
    ax1 = pl.subplot(gs[0, 0])
    plotci(ax1, x, E, grl_experiments_regrets, GRL_COLOR, GRL_LABEL)
    plotci(ax1, x, E, egl_experiments_regrets, EGL_COLOR, EGL_LABEL)
    plotci(ax1, x, E, ucb_experiments_regrets, UCB_COLOR, UCB_LABEL)
    ax1.legend()
    ax1.set_title("Cumulative Regret")
    ax1.set_xlabel("t")
    ax1.set_ylabel("regret")

    # ----- Rewards -----
    ax2 = pl.subplot(gs[0, 1])
    ax2.plot(
        x,
        np.mean(np.cumsum(grl_experiments_rewards, axis=1), axis=0),
        color=GRL_COLOR,
        label=GRL_LABEL,
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
    ax2.set_yscale("log")
    ax2.set_title("Cumulative Rewards")
    ax2.set_xlabel("t")
    ax2.set_ylabel("reward")

    # ----- Arms Pulled -----
    ax4 = pl.subplot(gs[1, :])
    data = {
        "greedy": np.mean(grl_arms_pulled, axis=0),
        "e-greedy": np.mean(egl_arms_pulled, axis=0),
        "UCB": np.mean(ucb_arms_pulled, axis=0),
    }
    colors = [
        GRL_COLOR,
        EGL_COLOR,
        UCB_COLOR,
    ]
    bar_plot(ax4, data, colors=colors, total_width=0.8, single_width=0.9)
    ax4.set_yscale("log")
    ax4.set_title("Arm pulls")
    ax4.set_xlabel("Arms")
    ax4.set_ylabel("Times Pulled")
    ax4.set_xticks([*range(n_arms)])
    ax4.set_xticklabels([f"arm {a}" for a in range(n_arms)])

    pl.show()
