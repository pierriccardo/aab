from libmab.learners import Learner, CombinatorialLearner
from abc import abstractclassmethod

import numpy as np


class Attacker(Learner):
    def __init__(self, n_arms: int, T: int, target: int) -> None:
        super().__init__(n_arms, T)
        self.target = target

    @abstractclassmethod
    def attack(self, reward, arm):
        pass


class EpsilonGreedyAttacker(Attacker):
    def __init__(self, n_arms: int, T: int, target: int, var: float = 1, delta: float = 0.05) -> None:
        super().__init__(n_arms, T, target)
        self.var = var
        self.delta = delta

    def attack(self, reward: float, arm: int) -> float:
        mu_i_old = self.estimates[arm]
        N_i_old = self.arm_pulls[arm]
        self.update(reward, arm)

        if arm == self.target:
            alpha = 0
        else:
            mu_k_new = self.estimates[self.target]
            N_i_new = self.arm_pulls[arm]
            N_k_new = self.arm_pulls[self.target]

            alpha = (
                (mu_i_old * N_i_old)
                + reward
                - ((mu_k_new - 2 * self.beta(N_k_new)) * N_i_new)
            )
            alpha = 0 if alpha is np.nan else max(0, alpha)

        self.attacks.append(alpha)
        return alpha


class ACEAttacker(Attacker):
    """Adaptive attack by Constant Estimation (ACE)
    paper: http://arxiv.org/abs/1905.06494
    """
    def __init__(self, n_arms: int, T: int, target: int, var: float = 1.0, delta=0.05):
        super().__init__(n_arms, T, target)
        self.var = var  # variance
        self.delta = delta  # high-prob. delta

    def beta(self, arm_pulls: int) -> float:
        return np.sqrt((2 * self.var / arm_pulls) * np.log(np.pi**2 * self.n_arms * arm_pulls**2 / (3 * self.delta)))

    def attack(self, reward, arm):
        # update inner Learner state
        self.update(reward, arm)

        corruption = 0
        if arm != self.target:
            N_i_new = self.arm_pulls[arm]
            N_k_new = self.arm_pulls[self.target]

            corruption = max(0, self.estimates[arm] - self.estimates[self.target] + self.beta(N_i_new) + self.beta(N_k_new))
            # corruption = 0 if corruption is np.nan else max(0, corruption)

        return corruption


class UCBJunAttacker(Attacker):
    def __init__(self, n_arms, T: int, target, var, delta=.05, delta0=1):
        super().__init__(n_arms, T, target)
        self.delta = delta
        self.var = var
        self.delta0 = delta0

        self.corruption_per_arm = [0.0 for _ in range(n_arms)]

    def beta(self, arm_pulls) -> float:
        return np.sqrt((2 * self.var / arm_pulls) * np.log(np.pi**2 * self.n_arms * arm_pulls**2 / (3 * self.delta)))

    def attack(self, reward, arm):
        self.update(reward, arm)
        if arm == self.target:
            return 0
        corruption = self.arm_pulls[arm] * self.estimates[arm]
        corruption += - self.corruption_per_arm[arm]
        corruption += - self.arm_pulls[arm] * (self.estimates[self.target] - 2*self.beta(self.arm_pulls[self.target]) - self.delta0)
        corruption = max(0, corruption)
        self.corruption_per_arm[arm] += corruption
        return corruption


class OracleAttacker(Attacker):

    def __init__(self, n_arms: int, T: int, target: int, means: [float], epsilon: float = .05):
        super().__init__(n_arms, T, target)
        self.means = means
        self.epsilon = epsilon

    def attack(self, reward: float, arm: int) -> float:
        if arm == self.target:
            return 0
        corruption = self.means[arm] - self.means[self.target] + self.epsilon
        return corruption

class CombinatorialAttacker(CombinatorialLearner):
    def __init__(self, n_arms: int, T: int, target: np.ndarray) -> None:
        super().__init__(n_arms, T)
        self.target = target

    @abstractclassmethod
    def attack(self, reward: np.ndarray, arm: np.ndarray) -> np.ndarray:
        pass

class OracleCombinatorialAttacker(CombinatorialAttacker):

    def __init__(self, n_arms: int, T: int, target: np.ndarray, means: np.ndarray, epsilon: float = .05) -> None:
        super().__init__(n_arms, T, target)
        self.means = means
        self.epsilon = epsilon

        self.lowest_target_mean = np.min(self.means[self.target == 1])
        self.len_target = len(self.target[self.target == 1])
        self.gap = self.lowest_target_mean / (self.n_arms - self.len_target + 1)


    def attack(self, reward: np.ndarray, arm: np.ndarray) -> np.ndarray:
        corruption = np.array([self.gap - self.epsilon for _ in range(self.n_arms)])
        corruption = self.means - corruption
        # here the ^1 is to negate the target, to attack only arms
        # that are not in the target arm
        corruption = corruption * (self.target ^ 1) * arm

        return corruption


