""" Maximum walk Q-learning for grid world, we require it converges to the optimal
policy and optimal value function.

Jingyu Liu, June 20, 2023.

"""

import sys
import random
import numpy as np

sys.path.append("../../src/")
from GridWorld import GridWorld

# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

epsilon = 0.1
step_size = 0.1
discount = 0.9
maxwalk = 10

GWopt = GridWorld()
GWopt.OptimalQ(discount=discount, tol=1e-4)

GW = GridWorld()
update_iter_number, episode_iter_number = (
    GW.QLearningMaxWalk(epsilon=epsilon,
                        step_size=step_size,
                        discount=discount,
                        maxwalk=maxwalk,
                        target_policy=GWopt.policy_))


policy_save_path = "../../images/qlearning_maxwalk_optimal_policy.png"
GW.ShowCurrentPolicy(save_file=policy_save_path)

policy_save_path = "../../images/qlearning_maxwalk_optimal_policy.eps"
GW.ShowCurrentPolicy(save_file=policy_save_path)

V_save_path = "../../images/qlearning_maxwalk_optimal_value.png"
GW.ShowCurrentV(V_save_path)

V_save_path = "../../images/qlearning_maxwalk_optimal_value.eps"
GW.ShowCurrentV(V_save_path)

state_update_number_save_path = (
    "../../images/qlearning_maxwalk_optimal_state_update_number.png")
GW.ShowCurrentStateUpdateNumber(state_update_number_save_path)

state_update_number_save_path = (
    "../../images/qlearning_maxwalk_optimal_state_update_number.eps")
GW.ShowCurrentStateUpdateNumber(state_update_number_save_path)

print(f"Update iter number in Q-learning episode: {update_iter_number}")

print(f"Episode iter number in Q-learning episode: {episode_iter_number}")