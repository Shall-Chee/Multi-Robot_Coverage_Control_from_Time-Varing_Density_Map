import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
import shutil

from param import *
from helpers import *
from cvt import CVT


class RobotEnv:
    def __init__(self, distribution, init_pos):
        self.distribution = distribution
        self.robot_cnt = len(init_pos)
        self.init_pos = init_pos

        self.cvt = CVT(self.distribution, self.robot_cnt, self.init_pos)
        self.prev_state = self.init_pos.copy()
        self.state = self.init_pos.copy()
        self.global_flag = True
        self.timestep = 0

    def get_action(self, target):
        action = Kp * (target - self.state) + Kd * ((target - self.state) - (self.state - self.prev_state))
        return action

    def get_target(self):
        # Local Planner: Towards Centroid
        local_error = np.sqrt(np.sum((self.cvt.centroids - self.state) ** 2, axis=1))
        local_planner = interp(self.state, self.cvt.centroids,
                               (local_move_limit / (local_error + eps)).clip(max=1)) - self.state

        # Global Planner: Better Allocate Cost
        cost = self.cvt.compute_h(self.cvt.regions, self.cvt.vertices, self.cvt.centroids)
        cost_order = np.argsort(cost)
        min_ind, max_ind = cost_order[0], cost_order[-1]
        if cost[min_ind] / cost[max_ind] > cost_ratio_thresh: self.global_flag = False  # finish reallocation
        global_error = np.sqrt(np.sum((self.state[max_ind] - self.state[min_ind]) ** 2))
        global_planner = np.zeros_like(self.state)
        global_planner[min_ind] = interp(self.state[min_ind], self.state[max_ind],
                                         (global_move_limit / (global_error + eps)).clip(max=1)) - self.state[min_ind]

        # Get Total Action and Step
        target = local_scale * local_planner + self.global_flag * global_scale * global_planner + self.state

        return target

    def render(self):
        voronoi_plot_2d(self.cvt.vor, ax=ax)
        for line in self.cvt.new_lines:
            plt.plot(line[0], line[1], 'k')
        plt.imshow(self.cvt.distribution)
        plt.gca().invert_yaxis()
        plt.xlim(X_MIN - X_SIZE * plot_padding, X_MAX + X_SIZE * plot_padding)
        plt.ylim(Y_MIN - Y_SIZE * plot_padding, Y_MAX + Y_SIZE * plot_padding)
        plt.plot(self.cvt.vertices[:, 0], self.cvt.vertices[:, 1], 'bo', label='vertices')
        plt.plot(self.cvt.centroids[:, 0], self.cvt.centroids[:, 1], 'go', label='centroids')
        plt.plot(self.prev_state[:, 0], self.prev_state[:, 1], 'co', label='prev_robot_position')
        plt.plot(self.state[:, 0], self.state[:, 1], 'ro', label='robot_position')
        plt.quiver(self.state[:, 0], self.state[:, 1],
                   self.state[:, 0] - self.prev_state[:, 0],
                   self.state[:, 1] - self.prev_state[:, 1], angles='xy', scale_units='xy', scale=1)
        plt.legend(loc="upper left")
        plt.title(f'number of robots = {self.robot_cnt}')
        if make_gif: gif_maker('coverage_control.gif', fig_dir, self.timestep, done, dpi=200)
        plt.pause(0.1)
        plt.cla()

    def reset(self):
        self.cvt = CVT(self.distribution, self.robot_cnt, self.init_pos)
        self.prev_state = self.init_pos.copy()
        self.state = self.init_pos.copy()
        self.global_flag = True
        self.timestep = 0

    def step(self, action):
        self.prev_state = self.state.copy()
        self.state += action
        self.state[:, 0] = self.state[:, 0].clip(X_MIN, X_MAX)
        self.state[:, 1] = self.state[:, 1].clip(Y_MIN, Y_MAX)
        self.timestep += 1
        if self.timestep % vor_duration == 0: self.cvt.step(self.state.copy())
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

    # Render
    fig = plt.figure()
    ax = plt.gca()
    show_animation = True
    make_gif = True

    # Save
    fig_dir = "./fig/"
    if os.path.exists(fig_dir): shutil.rmtree(fig_dir)

    # Simulate
    while not done:
        if env.timestep % vor_duration == 0: target = env.get_target()
        action = env.get_action(target)
        done = env.step(action)
        if show_animation: env.render()
        print("Timestep: {}  Error: {:.4f}".format(env.timestep, np.linalg.norm(action)))
