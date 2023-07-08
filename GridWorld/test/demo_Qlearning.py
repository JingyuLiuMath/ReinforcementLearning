""" Q-learning for grid world.

Jingyu Liu, June 19, 2023.

"""

import sys
import random
import numpy as np

sys.path.append("../src/")
from GridWorld import GridWorld

# For the same result.
seed = 1
random.seed(seed)
np.random.seed(seed)

epsilon = 0.1
step_size = 0.1
discount = 0.9
moving_step_number = 10
episode_number = 10000

GW = GridWorld()

GW.QLearning(epsilon=epsilon,
             step_size=step_size,
             discount=discount,
             moving_step_number=moving_step_number)

GW.ShowCurrentPolicy()

GW.ShowCurrentV()
