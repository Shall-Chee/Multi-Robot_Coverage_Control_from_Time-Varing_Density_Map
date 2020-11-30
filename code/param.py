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

DUMMY_CORNER = np.array([[999, 999], [-999, 999], [999, -999], [-999, -999]])

DENSITY_MIN = 0.01
DENSITY_MAX = 255

eps = 1e-3  # a small number to prevent divide by zero

# Simulation Parameters
robot_cnt = 6
max_timestep = 200
pos_error_thresh = 0.1  # reach stability if position change smaller than this
vor_duration = 5

# Kinematics
local_move_limit = 30
global_move_limit = 100
local_scale = 0.5
global_scale = 0.5
cost_ratio_thresh = 0.2

# Control
Kp = 0.25
Kd = -0.1
