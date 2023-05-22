""" Performance on long episodes of the Example 6.7 `Maximization Bias Example`
in Sutton's book `Reinforcement Learning: An introduction`.

Jingyu Liu, April 1, 2023.

"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

sys.path.append("../src/")
from MaxBias import MaxBias

# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

print("")
print("the performance of `Maximization Bias Example` on a long episode")

arm_number = 10
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

print("")
print(f"epsilon parameter for epsilon-greedy policy: {epsilon}")
print(f"step size for learning: {step_size}")
print(f"discout: {discount}")

total_runs = 10000
episode_number = 2000

print("")
print(f"number of total runs: {total_runs}")

left_action_q_array = np.zeros(episode_number)
left_action_qq_array = np.zeros(episode_number)
left_total_rate_q_array = np.zeros(episode_number)
left_total_rate_qq_array = np.zeros(episode_number)

for each_run in range(total_runs):
    left_action_array_q_one_run = (
        MB.QLearning(epsilon=epsilon,
                     step_size=step_size,
                     discount=discount,
                     episode_number=episode_number))
    left_action_q_array = np.add(
        left_action_q_array, left_action_array_q_one_run)
    left_action_array_qq_one_run = (
        MB.DoubleQLearning(epsilon=epsilon,
                           step_size=step_size,
                           discount=discount,
                           episode_number=episode_number))
    left_action_qq_array = np.add(
        left_action_qq_array, left_action_array_qq_one_run)

left_total_rate_q_array = left_action_q_array / total_runs
left_total_rate_qq_array = left_action_qq_array / total_runs

print(f"the last value of left_total_rate_q_array: "
      f"{left_total_rate_q_array[-1]}")
print(f"the last value of left_total_rate_qq_array: "
      f"{left_total_rate_qq_array[-1]}")

episodes = np.arange(1, episode_number + 1)
optimal_array = np.ones(episode_number) * epsilon / 2
fig = plt.figure()
plt.plot(episodes, left_total_rate_q_array,
         color='r', label='Q-learning')
plt.plot(episodes, left_total_rate_qq_array,
         color='g', label='Double Q-learning')
plt.plot(episodes, optimal_array, color='k',
         linestyle='--', label='optimal')
# xticks = list(range(0, total_episodes, 100))
# xticks[0] = 1
# plt.xticks(xticks)
plt.xlabel('Episodes')
plt.yticks([0.00, 0.05, 0.25, 0.50, 0.75, 1.00])
plt.ylabel('% left actions from A')
plt.legend()
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.show()
fig.savefig('../images/maximization_bias_long_episode_2000.eps')
fig.savefig('../images/maximization_bias_long_episode_2000.png')
