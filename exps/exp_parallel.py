from libmab.envs import GaussianEnv, BernoulliEnv
from libmab.learners import UCB, EpsilonGreedy, Greedy
from multiprocessing import Pool, Process
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

E = 10
T = 10**5
arms = [.2, .5, .4, .6, .23, .34]
K = len(arms)
sigma = 1.1

# ------------------------------
# Parameters
# ------------------------------

env = GaussianEnv(arms, sigma)
opt_arm = arms[env.opt_arm()]

instances = [
    [UCB, [K, T, sigma]],
    [EpsilonGreedy, [K, T]],
    [Greedy, [K, T]],
]

B = len(instances)
regrets = np.zeros((B, E, T))
rewards = np.zeros((B, E, T))
armpull = np.zeros((B, E, K))


def trial(e, regrets, rewards, armpull):
    bandits = [classname(*args) for (classname, args) in instances]
    for t in tqdm(range(T)):
        for b_id, bandit in enumerate(bandits):
            arm = bandit.pull_arm()
            reward = env.reward(arm, t, e)
            bandit.update(reward, arm)

            #  update data for visualization
            rewards[b_id, e, t] = reward
            regrets[b_id, e, t] = opt_arm - arms[arm]
            armpull[b_id, e, arm] += 1
    for b in bandits:
        print("="*100)
        print(" "*45 + f"E = {e}" + " " * 45)
        print("="*100)
        print(b)


if __name__ == "__main__":

    for e in range(E):
        p = Process(target=trial, args=(e, regrets, rewards, armpull))
        p.start()
        p.join()

    print(regrets)
    print(rewards)

    x = [*range(T)]

    # ----- Regrets -----
    fig, ax = plt.subplots()
    for b_id, (bandit, _) in enumerate(instances):
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
    for b_id, (bandit, _) in enumerate(instances):
        label = bandit.__class__.__name__
        y = np.mean(np.cumsum(rewards, axis=2), axis=1)[b_id]
        ax.plot(x, y, label=label)
    ax.legend()
    ax.set_title("Cumulative Rewards")
    ax.set_xlabel("t")
    ax.set_ylabel("Reward")
    ax.grid(True, ls='--', lw=.5)
    plt.show()


"""
import multiprocessing
from tqdm import tqdm
import numpy as np

# Define the number of experiments (E) and times (T)
E = 10
T = 100

# Example function to fill the rewards array for a specific experiment and time
def fill_rewards(experiment, time, rewards_array, lock):
    # Simulate some computation to get the reward value
    reward_value = experiment * time

    # Lock to prevent race conditions when updating the shared array
    with lock:
        # Assign the reward value to the corresponding position in the array
        rewards_array[experiment, time] = reward_value

# Function to fill the rewards array for all experiments and times
def fill_rewards_concurrently(rewards_array):
    # Create a multiprocessing manager to create a shared array
    with multiprocessing.Manager() as manager:
        # Create a shared array using the manager
        shared_rewards_array = manager.Array('d', rewards_array.flatten())

        # Reshape the shared array back to the original shape
        shared_rewards_array = np.frombuffer(shared_rewards_array).reshape(rewards_array.shape)

        # Use a multiprocessing Lock to synchronize access to the shared array
        lock = manager.Lock()

        # Create a list of processes
        processes = []

        # Iterate over experiments and times to create and start processes
        for e in range(E):
            for t in range(T):
                process = multiprocessing.Process(target=fill_rewards, args=(e, t, shared_rewards_array, lock))
                processes.append(process)
                process.start()

        # Wait for all processes to complete
        for process in processes:
            process.join()

    # Copy the shared array back to the original array
    np.copyto(rewards_array, shared_rewards_array)

# Create an empty rewards array
rewards = np.zeros((E, T))

# Fill the rewards array concurrently using multiprocessing
fill_rewards_concurrently(rewards)

# Print the filled rewards array
print(rewards)

"""