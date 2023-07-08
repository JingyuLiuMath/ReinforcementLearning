""" value function of E-SARSA for grid world after 100w updates.

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
moving_step_number = 1000000

GW = GridWorld()
GW.ESARSA(epsilon=epsilon,
          step_size=step_size,
          discount=discount,
          moving_step_number=moving_step_number)

V_save_path = "../../images/esarsa_value_after_100w_update.png"
GW.ShowCurrentV(V_save_path)

V_save_path = "../../images/esarsa_value_after_100w_update.eps"
GW.ShowCurrentV(V_save_path)
