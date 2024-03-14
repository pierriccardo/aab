from collections import defaultdict

import numpy as np
import abc


class CombinatorialEnv():

    def __init__(self, arms: np.ndarray, d: int = None) -> None:
        self.arms = arms
        self.K = len(arms)
        self.d = self.K if d is None else d

    @abc.abstractclassmethod
    def reward(self, arm: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def rewardvec(self, e: int, t: int, seed: int = 0) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def pseudo_reward(self, arm: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def opt_arm(self) -> np.ndarray:
        pass

    def __str__(self):
        return f"""
            Env: {self.__class__.__name__}
            K       : {self.K}
            opt arm : {self.opt_arm()}
        """

class CombinatorialGaussianEnv(CombinatorialEnv):

    def __init__(self, arms: np.ndarray, sigma: float = .1, d: int = None) -> None:
        super().__init__(arms)
        self.sigma = sigma
        self.d = self.K if d is None else d

    def reward(self, arm: np.array) -> np.array:
        return np.random.normal(self.arms, self.sigma) * arm

    def rewardvec(self, e: int, t: int, seed: int = 0) -> np.ndarray:
        """When comparing different algorithms on the same environment
        it is important to ensure that are compared on the same instance
        generating the reward vector beforehand, seeded with exp. number
        and current round can be seen as precomputing the whole reward
        table (E x T) and draw at each round the correct vector

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
        List[float]
            Reward realization vector for exp. num e and
            round t.
        """

        # to achieve a repeatable random state we double
        # seed the generator with experiment and round num
        rng = np.random.default_rng(hash((e, t)) & 0xFFFFFFFF)

        return rng.normal(self.arms, self.sigma)

    def pseudo_reward(self, arm: np.array) -> np.array:
        return self.arms * arm

    def opt_arm(self) -> np.ndarray:
        opt = np.zeros(self.K)
        # find indexes of d maximum elements
        idx = np.argpartition(self.arms, -self.d)[-self.d:]
        opt[idx] = 1
        return opt


class PMCEnv():

    """Probabilistic Maximum Coverage (PMC)
    This environment simulate a PMC problem where given a bipartite graph G(L, R, E) a learner
    must find in an online fashion the subset S subseteq L of the d elements that maximize
    the activation of vertices in R.
    """

    def __init__(self, bgraph: dict,  d: int = None) -> None:
        # TODO: refactor graph class to handle bot graph for online-shortest-path
        # and bipartite graph.
        # TODO: consider using networkx

        # implementation follows from:
        # https://jmlr.org/papers/volume17/14-298/14-298.pdf

        self.L = bgraph.keys()
        self.E = defaultdict(list)
        self.P = defaultdict(list)
        for source, edges in bgraph.items():
            for target, p in edges.items():
                self.E[source].append(target)
                self.P[source].append(p)

        self.d = len(self.L) - 1 if d is None else d

        self.K = len(self.L)  # n_arms

    def reward(self, arm: np.array) -> np.array:
        # arm is an array {0, 1}^|L| specifing the
        # vertices selected by the learner on L
        # the reward it the bernoulli realisation
        # of each edge, thus
        reward = np.zeros(len(self.L))

        # each arm is a subset S \subseteq L,
        # and every element of S = {0, 1}^|L|
        # specifies wheter the vertex v \in L
        # has been selected or not. For each
        # edge e = (v, *) a bernoulli reward is sampled
        # with probability p(v, u)
        for idx, v in enumerate(self.L):
            for p in self.P[v]:
                #reward[idx] += np.random.binomial(1, p)
                reward[idx] += np.random.normal(p, 2)
        return reward * arm

    def opt_arm(self):
        means = np.zeros(len(self.L))
        for idx, v in enumerate(self.L):
            means[idx] = np.sum(self.P[v])

        opt = np.zeros(len(self.L))
        # find indexes of d maximum elements
        idx = np.argpartition(means, -self.d)[-self.d:]
        opt[idx] = 1
        return opt

    def opt_reward(self):
        return self.pseudo_reward(self.opt_arm())

    def pseudo_reward(self, arm):
        # return pseudo reward, i.e. the reward
        # knowing the true means
        reward = np.zeros(len(self.L))
        for idx, v in enumerate(self.L):
            reward[idx] = sum(self.P[v]) * len(self.P[v])
        return reward * arm

    def __str__(self):
        return f"""
            Env: {self.__class__.__name__}
            K       : {self.K}
            opt arm : {self.opt_arm()}
        """


class Graph:
    # DAG

    def __init__(self, vertices) -> None:
        self.V = vertices
        self.E = defaultdict(list)
        self.W = defaultdict(list)

    def add_edge(self, s, t, w=0):
        self.E[s].append(t)  # edges
        self.W[s].append(w)  # weights

    def shortest_path(self):
        pass

    def _find_paths_rec(self, u, d, paths, visited, path=[]):
        # DFS to find all the paths, the graph must be acyclic
        visited[u] = True
        path.append(u)

        if u == d:
            paths.append(copy.copy(path))
        else:
            for i in self.E[u]:
                if visited[i] == False:
                    self._find_paths_rec(i, d, paths, visited, path=path)
        path.pop()
        visited[u] = False

    def find_paths(self, s, t):
        paths = []
        visited = {i: False for i in self.V}
        self.find_paths_rec(s, t, paths, visited)
        return paths

    def is_cyclic(self):
        # TODO: implement cycles controller
        return True

    def __str__(self) -> str:
        s = f"Vertices: {self.V}\n"
        s += "Edges:\n"
        for v in self.V:
            for e, w in zip(self.E[v], self.W[v]):
                s += f"{v} --[{w}]--> {e}\n"
        return s


class OnlineShortestPathEnv():

    def __init__(self) -> None:
        # TODO: to be implemented
        pass

