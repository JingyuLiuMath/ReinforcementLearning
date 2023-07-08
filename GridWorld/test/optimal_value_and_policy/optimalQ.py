""" Optimal Q function for grid world.

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

discount = 0.9

GW = GridWorld()

sweep_iter_number = GW.OptimalQ(discount=discount, tol=1e-4)

policy_save_file = "../../images/optimal_policy.png"
GW.ShowCurrentPolicy(save_file=policy_save_file)

policy_save_file = "../../images/optimal_policy.eps"
GW.ShowCurrentPolicy(save_file=policy_save_file)

print(f"Sweep iter number in q function iteration: {sweep_iter_number}")
