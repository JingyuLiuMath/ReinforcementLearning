""" Optimal value function for grid world.

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

sweep_iter_number = GW.OptimalV(discount=discount, tol=1e-4)

value_save_file = "../../images/optimal_value.png"
GW.ShowCurrentV(save_file=value_save_file)

value_save_file = "../../images/optimal_value.eps"
GW.ShowCurrentV(save_file=value_save_file)

print(f"Sweep iter number in value function iteration: {sweep_iter_number}")
