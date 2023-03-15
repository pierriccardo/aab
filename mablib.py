import numpy as np

class Agent:

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.t = 0
        self.arm_pulls = np.zeros(n_arms)
        self.estimates = np.zeros(n_arms)
        self.rewards = []
    
    def update(self, reward, arm):
        
        self.rewards.append(reward)
        self.estimates[arm] = (self.estimates[arm] * self.arm_pulls[arm] + reward) / (self.arm_pulls[arm] + 1)
        self.arm_pulls[arm] += 1
        self.t+=1
        
    def pull_arm(self):
        pass

class Greedy(Agent):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        else:
            return np.argmax(self.estimates)

class EpsilonGreedy(Agent):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        else:
            if np.random.random() <= 1/self.t:
                return np.random.choice(range(self.n_arms))
            else:
                return np.argmax(self.estimates)

class UCBClassic(Agent):

    def __init__(self, n_arms: int, c: int = 2):
        super().__init__(n_arms)
        self.c = c

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        else:
            exploration = self.c * np.log(self.t) / self.arm_pulls
            exploration = np.sqrt(exploration)
            sel = np.add(self.estimates,exploration)
            return np.argmax(sel)

class UCB(Agent):

    def __init__(self, n_arms: int, c: int = 1, sigma: float = .1):
        super().__init__(n_arms)
        self.c = c
        self.sigma = sigma

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        else:
            exploration = self.c * np.log(self.t) / self.arm_pulls
            exploration = np.sqrt(exploration) * 3 * self.sigma
            sel = np.add(self.estimates,exploration)
            return np.argmax(sel)

class Attacker(Agent):

    def __init__(self, n_arms, target, var, delta=.1):
        """
        Var is the variance which is supposed to be known
        """
        super().__init__(n_arms)
        self.target = target
        self.var = var
        self.delta = delta
        self.attacks = []
        
    def beta(self, arm_pulls):
        arg = (2 * self.var / arm_pulls) * (np.log(np.pi**2 * self.n_arms * arm_pulls**2 / (3 * self.delta)))
        return np.sqrt(arg)
    
    def attack(self, reward, arm):
        return 0
        

class EpsilonGreedyAttacker(Attacker):

    def __init__(self, n_arms, target, var, delta=.1):
        super().__init__(n_arms, target, var, delta)

    def attack(self, reward, arm):
        mu_i_old = self.estimates[arm]
        N_i_old  = self.arm_pulls[arm]
        self.update(reward, arm)

        if arm == self.target:
            alpha = 0
        else:
            mu_k_new = self.estimates[self.target]
            N_i_new  = self.arm_pulls[arm]
            N_k_new  = self.arm_pulls[self.target]
            
            alpha = (mu_i_old * N_i_old) + reward - ((mu_k_new - 2*self.beta(N_k_new)) * N_i_new)
            alpha = 0 if alpha is np.nan else max(0, alpha)

        self.attacks.append(alpha)
        return alpha

class ACEAttacker(Attacker):

    def __init__(self, n_arms, target, var, delta=.1):
        super().__init__(n_arms, target, var, delta)
    
    def attack(self, reward, arm):
        self.update(reward, arm)
        if arm == self.target:
            alpha = 0
        else:
            mu_i_new = self.estimates[arm]
            mu_k_new = self.estimates[self.target]
            N_i_new  = self.arm_pulls[arm]
            N_k_new  = self.arm_pulls[self.target]

            alpha = mu_i_new - mu_k_new + self.beta(N_i_new) + self.beta(N_k_new)
            alpha = 0 if alpha is np.nan else max(0, alpha)

        self.attacks.append(alpha)
        return alpha