""" Q-learning for grid world without ergodicity, we require it converges to the
optimal policy and optimal value function.

Jingyu Liu, June 19, 2023.

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

GWopt = GridWorld()
GWopt.OptimalQ(discount=discount, tol=1e-4)

GW = GridWorld()
update_iter_number = GW.QLearning(epsilon=epsilon,
                                  step_size=step_size,
                                  discount=discount,
                                  target_policy=GWopt.policy_)


policy_save_path = "../../images/qlearning_optimal_policy.png"
GW.ShowCurrentPolicy(save_file=policy_save_path)

policy_save_path = "../../images/qlearning_optimal_policy.eps"
GW.ShowCurrentPolicy(save_file=policy_save_path)

V_save_path = "../../images/qlearning_optimal_value.png"
GW.ShowCurrentV(V_save_path)

V_save_path = "../../images/qlearning_optimal_value.eps"
GW.ShowCurrentV(V_save_path)

state_update_number_save_path = (
    "../../images/qlearning_optimal_state_update_number.png")
GW.ShowCurrentStateUpdateNumber(state_update_number_save_path)

state_update_number_save_path = (
    "../../images/qlearning_optimal_state_update_number.eps")
GW.ShowCurrentStateUpdateNumber(state_update_number_save_path)

print(f"Update iter number in Q-learning: {update_iter_number}")
