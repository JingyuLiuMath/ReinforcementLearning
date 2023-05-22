""" Show how the class MaxBias works.

Jingyu Liu, May 22, 2023.

"""

import sys
import random
import numpy as np

sys.path.append("../src/")
from MaxBias import MaxBias

# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

print("")
print("This is a simple test file to show how the class MaxBias works.")

arm_number = 3
mu = -0.1
sigma = 1.0
MB = MaxBias(arm_number=arm_number, mu=mu, sigma=sigma)

print("")
print(f"arm number of the multi-arm bandit: {arm_number}")
print(f"expection of the multi-arm bandit: {mu}")
print(f"standard deviation of the multi-arm bandit: {sigma}")

epsilon = 0.1
step_size = 0.1
discount = 1.0
episode_number = 300

print("")
print(f"epsilon parameter for epsilon-greedy policy: {epsilon}")
print(f"step size for learning: {step_size}")
print(f"discout: {discount}")
print(f"number of episodes: {episode_number}")

left_action_list_q = (
    MB.QLearning(epsilon=epsilon,
                 step_size=step_size,
                 discount=discount,
                 episode_number=episode_number))
print("")
print(f"Successfully finish Q-learning!")

left_action_list_qq = (MB.DoubleQLearning(epsilon=epsilon,
                                          step_size=step_size,
                                          discount=discount,
                                          episode_number=episode_number))
print("")
print(f"Successfully finish double Q-learning!")
