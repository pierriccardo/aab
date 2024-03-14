from collections import defaultdict
from typing import List
import numpy as np
import hashlib
import copy
import abc

class Env():

    def __init__(self, arms: List[float]) -> None:
        self.arms = arms
        self.K = len(arms)

    @abc.abstractclassmethod
    def reward(self):
        pass

    @abc.abstractclassmethod
    def rewardvec(self):
        pass

    @abc.abstractclassmethod
    def pseudo_reward(self):
        pass

    @abc.abstractclassmethod
    def opt_arm(self):
        pass

    def __str__(self):
        return f"""
            Env: {self.__class__.__name__}
            K       : {self.K}
            opt arm : {self.opt_arm()}
        """


class GaussianEnv(Env):

    def __init__(self, arms: List[float], sigma: float = .1) -> None:
        super().__init__(arms)
        self.sigma = sigma

    def reward(self, arm: int, t: int, e: int = 0) -> float:
        return np.random.normal(self.arms[arm], self.sigma)

    def rewardvec(self, e: int, t: int, seed: int = 0) -> np.ndarray:
        """When comparing different algorithms on the same environment
        it is important to ensure that are compared on the same instance
        generating the reward vector beforehand, seeded with exp. number
        and current round can be seen as precomputing the whole reward
        table (E x T) and draw at each round the correct vector.

        Parameters
        ----------
        e : int
            current experiment number e in E
        t : int
            current round number t in T
        seed : int, optional
            an additional seed, by default 0

        Returns
        -------
        np.ndarray
            Reward realization vector for exp. num e and
            round t.
        """

        # to achieve a repeatable random state we double
        # seed the generator with experiment and round num
        rng = np.random.default_rng(hash((e, t)) & 0xFFFFFFFF)

        return rng.normal(self.arms, self.sigma)

    def pseudo_reward(self, arm: int) -> float:
        return self.arms[arm]

    def opt_arm(self):
        return np.argmax(self.arms)


class BernoulliEnv(Env):

    def __init__(self, arms: [float]) -> None:
        super().__init__(arms)

    def reward(self, arm):
        return 1 if np.random.random() < self.arms[arm] else 0

