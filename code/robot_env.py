import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

from param import *
from helpers import *
from cvt import CVT


class RobotEnv:
    def __init__(self, distribution, init_pos):
        self.distribution = distribution
        self.robot_cnt = len(init_pos)
        self.init_pos = init_pos

        self.cvt = CVT(self.distribution, self.robot_cnt, self.init_pos)
        self.state = self.init_pos.copy()
        self.global_flag = True
        self.timestep = 0

    def get_action(self, target):
        # Local Planner: Towards Centroid
        local_error = np.sqrt(np.sum((target - self.state) ** 2, axis=1))
        local_planner = interp(self.state, target, (local_move_limit / (local_error + eps)).clip(max=1)) - self.state

        # Global Planner: Better Allocate Cost
        cost = self.cvt.compute_h(self.cvt.regions, self.cvt.vertices, target)
        cost_order = np.argsort(cost)
        min_ind, max_ind = cost_order[0], cost_order[-1]
        if cost[min_ind] / cost[max_ind] > cost_ratio_thresh: self.global_flag = False  # finish reallocation
        global_error = np.sqrt(np.sum((self.state[max_ind] - self.state[min_ind]) ** 2))
        global_planner = np.zeros_like(self.state)
        global_planner[min_ind] = interp(self.state[min_ind], self.state[max_ind],
                                         (global_move_limit / (global_error + eps)).clip(max=1)) - self.state[min_ind]

        # Get Total Action and Step
        action = local_scale * local_planner + self.global_flag * global_scale * global_planner

        return action

    def get_target(self):
        return self.cvt.centroids

    def render(self):
        voronoi_plot_2d(self.cvt.vor)
        plt.imshow(self.cvt.distribution)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.plot((self.state - self.get_action(self.get_target()))[:, 0],
                 (self.state - self.get_action(self.get_target()))[:, 1], 'ro', label='robot_position')
        plt.xlim(X_MIN - X_SIZE * 0.1, X_MAX + X_SIZE * 0.1)
        plt.ylim(Y_MIN - Y_SIZE * 0.1, Y_MAX + Y_SIZE * 0.1)
        plt.plot(self.cvt.vertices[:, 0], self.cvt.vertices[:, 1], 'bo', label='vertices')
        plt.plot(self.cvt.centroids[:, 0], self.cvt.centroids[:, 1], 'go', label='centroids')
        plt.plot(self.state[:, 0], self.state[:, 1], 'co', label='new_robot_position')
        plt.quiver(self.state[:, 0], self.state[:, 1],
                   self.get_action(self.get_target())[:, 0],
                   self.get_action(self.get_target())[:, 1], angles='xy', scale_units='xy', scale=1)
        plt.legend(loc="upper left")
        plt.title(f'number of robots = {self.robot_cnt}')
        plt.show()

    def reset(self):
        self.cvt = CVT(self.distribution, self.robot_cnt, self.init_pos)
        self.state = self.init_pos.copy()
        self.global_flag = True
        self.timestep = 0

    def step(self, action):
        self.state += action
        self.timestep += 1
        self.cvt.step(self.state)
        done = np.linalg.norm(action) < pos_error_thresh or self.timestep > max_timestep

        return done


if __name__ == '__main__':
    # Read Distribution/Density Map
    distribution = np.load('./target_distribution.npy')
    scatter_ratio = np.random.random((robot_cnt, 2))
    robot_pos = interp([X_MIN, Y_MIN], [X_MAX, Y_MAX], scatter_ratio)  # generate robot on random initial positions

    # Initialize Environment
    env = RobotEnv(distribution, robot_pos)
    done = False

    # Simulate
    while not done:
        target = env.get_target()
        action = env.get_action(target)
        done = env.step(action)
        env.render()
        print("Timestep: {}  Error: {:.4f}".format(env.timestep, np.linalg.norm(action)))
