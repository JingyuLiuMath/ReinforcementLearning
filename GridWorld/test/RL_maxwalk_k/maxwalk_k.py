""" Maximum walk for grid world, we require it converges to the optimal policy
and optimal value function.

Jingyu Liu, June 20, 2023.

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
maxwalk = 20
run_number = 100

GWopt = GridWorld()
GWopt.OptimalQ(discount=discount, tol=1e-4)

maxwalk_array = np.arange(1, maxwalk + 1)

# SARSA
update_iter_number_sarsa_array = np.zeros(maxwalk)
for it in range(maxwalk):
    # For the same result.
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    for run_it in range(run_number):
        GW = GridWorld()
        update_iter_number, episode_iter_number = (
            GW.SARSAMaxWalk(epsilon=epsilon,
                            step_size=step_size,
                            discount=discount,
                            maxwalk=maxwalk_array[it],
                            target_policy=GWopt.policy_))
        update_iter_number_sarsa_array[it] += update_iter_number

update_iter_number_sarsa_array = (
        update_iter_number_sarsa_array / run_number)

# Q-learning
update_iter_number_qlearning_array = np.zeros(maxwalk)
for it in range(maxwalk):
    # For the same result.
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    for run_it in range(run_number):
        GW = GridWorld()
        update_iter_number, episode_iter_number = (
            GW.QLearningMaxWalk(epsilon=epsilon,
                                step_size=step_size,
                                discount=discount,
                                maxwalk=maxwalk_array[it],
                                target_policy=GWopt.policy_))
        update_iter_number_qlearning_array[it] += update_iter_number

update_iter_number_qlearning_array = (
        update_iter_number_qlearning_array / run_number)

# E-SARSA
update_iter_number_esarsa_array = np.zeros(maxwalk)
for it in range(maxwalk):
    # For the same result.
    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    for run_it in range(run_number):
        GW = GridWorld()
        update_iter_number, episode_iter_number = (
            GW.ESARSAMaxWalk(epsilon=epsilon,
                             step_size=step_size,
                             discount=discount,
                             maxwalk=maxwalk_array[it],
                             target_policy=GWopt.policy_))
        update_iter_number_esarsa_array[it] += update_iter_number

update_iter_number_esarsa_array = (
        update_iter_number_esarsa_array / run_number)

fig = plt.figure()
plt.plot(maxwalk_array, update_iter_number_sarsa_array,
         color='r', label='SARSA')
plt.plot(maxwalk_array, update_iter_number_qlearning_array,
         color='g', label='Q-learning')
plt.plot(maxwalk_array, update_iter_number_esarsa_array,
         color='b', label='E-SARSA')

xticks = list(range(1, maxwalk + 1))
plt.xticks(xticks)
plt.xlabel('Maximum walk number')
plt.ylabel('Update iter number')
plt.legend()
plt.show()
fig.savefig('../../images/maxwalk.png')
fig.savefig('../../images/maxwalk.eps')
