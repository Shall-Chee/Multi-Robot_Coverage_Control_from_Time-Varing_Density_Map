import numpy as np

# Random Seed
np.random.seed(0)

# Constant
X_MIN = 0
X_MAX = 127
Y_MIN = 0
Y_MAX = 127
X_SIZE = X_MAX - X_MIN
Y_SIZE = Y_MAX - Y_MIN

DENSITY_MIN = 0.01
DENSITY_MAX = 255

eps = 1e-3  # a small number to prevent divide by zero

# Simulation Parameters
robot_cnt = 6
max_timestep = 25
pos_error_thresh = 0.1  # reach stability if position change smaller than this

# Kinematics
local_move_limit = 10
global_move_limit = 20
local_scale = 1
global_scale = 1
cost_ratio_thresh = 0.1
