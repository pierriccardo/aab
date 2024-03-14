from tqdm import tqdm

import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import numpy as np
import itertools
import abc

np.set_printoptions(
    precision=3,
    suppress=True
)

class Learner:
    def __init__(self, n_arms: int, T: int) -> None:
        self.T = T  # time horizon
        self.n_arms = n_arms
        self.t = 0  # current time
        self.arm_pulls = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)
        self.rewards = np.zeros(T)

    def update(self, reward, arm) -> None:
        self.rewards[self.t] = reward
        self.estimates[arm] = (self.estimates[arm] * self.arm_pulls[arm] + reward) / (
            self.arm_pulls[arm] + 1
        )
        self.arm_pulls[arm] += 1
        self.t += 1

    def reset(self):
        self.t = 0
        self.arm_pulls = np.zeros(self.n_arms)
        self.estimates = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.T)

    def get_rewards(self):
        return self.rewards

    def get_arm_pulls(self):
        return self.arm_pulls

    def __str__(self) -> str:
        return f"""
            Learner: {self.__class__.__name__}
            T={self.T}, K={self.n_arms}
            arm_pulls  : {self.arm_pulls}
            estimates  : {self.estimates}
            tot reward : {np.sum(self.rewards)}
        """

    @abc.abstractclassmethod
    def pull_arm(self) -> int:
        pass


class Greedy(Learner):
    def __init__(self, n_arms: int, T: int) -> None:
        super().__init__(n_arms, T)

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        else:
            return np.argmax(self.estimates)


class EpsilonGreedy(Learner):
    def __init__(self, n_arms: int, T: int, epsilon: callable = lambda x: 1 / x):
        super().__init__(n_arms, T)
        self.epsilon = epsilon

    def pull_arm(self) -> int:
        if self.t < self.n_arms:
            return self.t
        else:
            if np.random.random() <= self.epsilon(self.t):
                return np.random.choice(range(self.n_arms))
            else:
                return np.argmax(self.estimates)


class UCB(Learner):
    def __init__(self, n_arms: int, T: int, sigma: float = 1) -> None:
        super().__init__(n_arms, T)
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


class CombinatorialLearner(Learner):

    def __init__(self, n_arms: int, T: int, d: int = None) -> None:
        super().__init__(n_arms, T)
        self.rewards = np.zeros((T, n_arms))
        self.d = n_arms if d is None else d

        assert self.d <= n_arms, "d cannot be >= n_arms"

    def update(self, reward: np.array, superarm: np.array) -> None:
        # reward    shape (n_arms,)
        # superarm  shape (n_arms,)
        # superarm is a vector {0, 1}^n_arms
        self.rewards[self.t, :] = reward
        self.estimates = (self.estimates * self.arm_pulls + reward) / (
            self.arm_pulls + superarm
        )
        self.estimates[np.isnan(self.estimates)] = 0
        self.arm_pulls += superarm
        self.t += 1

    def get_rewards(self) -> np.array:
        # return reward collected at each timestep
        # reward for a given timestep is the sum
        # of rewards for each basic arm
        return np.sum(self.rewards, axis=1)

    def reset(self):
        self.t = 0
        self.arm_pulls = np.zeros(self.n_arms)
        self.estimates = np.zeros(self.n_arms)
        self.rewards = np.zeros((self.T, self.n_arms))

    def __str__(self) -> str:

        return f"""
            Learner: {self.__class__.__name__}
            T={self.T}, K={self.n_arms}
            arm_pulls  : {self.arm_pulls}
            estimates  : {self.estimates}
            tot reward : {np.sum(self.get_rewards())}
        """


class CUCB(CombinatorialLearner):

    def __init__(self, n_arms: int, T: int, oracle: callable, sigma: float = .1, d: int = None) -> None:
        super().__init__(n_arms, T, d)
        self.oracle = oracle
        self.sigma = sigma

    def pull_arm(self):

        if self.t < self.n_arms:
            superarm = np.zeros(self.n_arms)
            indexes = np.random.randint(self.n_arms, size=self.d - 1)
            superarm[self.t] = 1
            superarm[indexes] = 1
            return superarm

        exploration = np.log(self.t) / self.arm_pulls
        exploration = 3 * self.sigma * np.sqrt(exploration)
        ucb_arms = np.add(self.estimates, exploration)

        return self.oracle(ucb_arms, self.d)


class Fixed(CombinatorialLearner):
    # TODO: refactor combinatorial class, create abstract
    # CombinatorialLearner(Learner)

    def __init__(self, n_arms: int, T: int, d: int = None, arm: np.array = None) -> None:
        super().__init__(n_arms, T, d)

        if arm is None:
            self.arm = np.zeros(n_arms)
            self.arm[0] = 1
        else:
            self.arm = arm

    def pull_arm(self):
        return self.arm


class CRandom(CombinatorialLearner):

    def __init__(self, n_arms: int, T: int, d: int = None) -> None:
        super().__init__(n_arms, T, d)

        #  arm instantiated here to avoid re-instatiate it
        #  on each pull_arm() call
        self.arm = np.zeros(self.n_arms)

    def pull_arm(self):
        tmp = np.random.randint(self.n_arms, size=np.random.randint(self.d))
        self.arm[:] = 0
        self.arm[tmp] = 1
        return self.arm


if __name__ == "__main__":
    T = 100000
    E = 10
    #arms = [0.49, 0.67, 0.35, 0.90]
    arms = np.random.normal(0, 1, 50)
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
        grl = Greedy(n_arms, T)
        egl = EpsilonGreedy(n_arms, T)
        ucb = UCB(n_arms, T)

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
    ax1 = pl.subplot(gs[0, :])
    ax1.plot(x, np.mean(np.cumsum(grl_experiments_regrets, axis=1), axis=0), color=GRL_COLOR, label=GRL_LABEL)
    ax1.plot(x, np.mean(np.cumsum(egl_experiments_regrets, axis=1), axis=0), color=EGL_COLOR, label=EGL_LABEL)
    ax1.plot(x, np.mean(np.cumsum(ucb_experiments_regrets, axis=1), axis=0), color=UCB_COLOR, label=UCB_LABEL)
    ax1.legend()
    ax1.set_title("Cumulative Regret")
    ax1.set_xlabel("t")
    ax1.grid(ls="--")
    ax1.set_ylabel("regret")

    # ----- Rewards -----
    ax2 = pl.subplot(gs[1, :])
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
    ax2.set_title("Cumulative Rewards")
    ax2.grid(ls="--")
    ax2.set_xlabel("t")
    ax2.set_ylabel("reward")
    pl.show()