""" Performance after 100w updates for grid world.

Jingyu Liu, June 26, 2023.

"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../src/")
from GridWorld import GridWorld

epsilon = 0.1
step_size = 0.1
discount = 0.9
moving_step_number = 1000000
observe_interval = 10000
run_number = 1

GWopt = GridWorld()
GWopt.OptimalQ(discount=discount, tol=1e-4)

moving_step_number_array = np.arange(0, moving_step_number + 1,
                                     observe_interval)

# SARSA
optimal_action_proportion_sarsa_array = np.zeros(
    len(moving_step_number_array))
# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

for run_it in range(run_number):
    GW = GridWorld()
    tmp_optimal_action_proportion_sarsa_array = (
        GW.SARSAObserve(target_policy=GWopt.policy_,
                        epsilon=epsilon,
                        step_size=step_size,
                        discount=discount,
                        moving_step_number=moving_step_number,
                        observe_interval=observe_interval))
    optimal_action_proportion_sarsa_array = (
            optimal_action_proportion_sarsa_array +
            tmp_optimal_action_proportion_sarsa_array)
optimal_action_proportion_sarsa_array = (
        optimal_action_proportion_sarsa_array / run_number)

# Q-learning
optimal_action_proportion_qlearning_array = np.zeros(
    len(moving_step_number_array))
# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

for run_it in range(run_number):
    GW = GridWorld()
    tmp_optimal_action_proportion_qlearning_array = (
        GW.QLearningObserve(target_policy=GWopt.policy_,
                            epsilon=epsilon,
                            step_size=step_size,
                            discount=discount,
                            moving_step_number=moving_step_number,
                            observe_interval=observe_interval))
    optimal_action_proportion_qlearning_array = (
            optimal_action_proportion_qlearning_array +
            tmp_optimal_action_proportion_qlearning_array)
optimal_action_proportion_qlearning_array = (
        optimal_action_proportion_qlearning_array / run_number)

# E-SARSA
optimal_action_proportion_esarsa_array = np.zeros(
    len(moving_step_number_array))
# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

for run_it in range(run_number):
    GW = GridWorld()
    tmp_optimal_action_proportion_esarsa_array = (
        GW.ESARSAObserve(target_policy=GWopt.policy_,
                         epsilon=epsilon,
                         step_size=step_size,
                         discount=discount,
                         moving_step_number=moving_step_number,
                         observe_interval=observe_interval))
    optimal_action_proportion_esarsa_array = (
            optimal_action_proportion_esarsa_array +
            tmp_optimal_action_proportion_esarsa_array)
optimal_action_proportion_esarsa_array = (
        optimal_action_proportion_esarsa_array / run_number)

fig = plt.figure()
plt.plot(moving_step_number_array,
         optimal_action_proportion_sarsa_array,
         color='r', label='SARSA')
plt.plot(moving_step_number_array,
         optimal_action_proportion_qlearning_array,
         color='g', label='Q-learning')
plt.plot(moving_step_number_array,
         optimal_action_proportion_esarsa_array,
         color='b', label='E-SARSA')

xticks = list(moving_step_number_array)
plt.xticks(xticks)
plt.xlabel('Moving step array')
plt.ylabel('Optimal action proportion')
plt.legend()
plt.show()
# fig.savefig('../../images/optimal_action_proportion_in_100w_update.png')
# fig.savefig('../../images/optimal_action_proportion_in_100w_update.eps')
