from scipy.spatial import Voronoi, voronoi_plot_2d
import shutil

from helpers import *
from cvt import CVT


class RobotEnv:
    def __init__(self, distribution, init_pos, policy="local"):
        self.policy = policy
        if self.policy == "local":
            self.distribution, self.local_scale, self.global_scale = distribution, 1.0, 0.0
        elif self.policy == "global":
            self.distribution, self.local_scale, self.global_scale = distribution, local_scale, global_scale
        elif policy == "exponential":
            self.distribution, self.local_scale, self.global_scale = exp_map(distribution), 1.0, 0.0

        self.robot_cnt = len(init_pos)
        self.init_pos = init_pos.copy()

        self.cvt = CVT(self.distribution, self.robot_cnt, self.init_pos)
        self.prev_state = self.init_pos.copy()
        self.state = self.init_pos.copy()
        self.path = []
        self.cost = []
        self.policy = policy
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

        # Local Planner: Collision Avoidance
        local_planner += collision_planner(self.state)

        # Global Planner: Better Allocate Cost
        cost = self.cvt.cost.copy()
        if not self.cost or cost.sum() < self.cost[-1]: self.cost.append(cost.sum())
        cost_order = np.argsort(cost)
        min_ind, max_ind = cost_order[0], cost_order[-1]
        if cost[min_ind] / cost[max_ind] > cost_ratio_thresh: self.global_flag = False  # finish reallocation
        global_error = np.sqrt(np.sum((self.state[max_ind] - self.state[min_ind]) ** 2))
        global_planner = np.zeros_like(self.state)
        global_planner[min_ind] = interp(self.state[min_ind], self.state[max_ind],
                                         (global_move_limit / (global_error + eps)).clip(max=1)) - self.state[min_ind]

        # Get Total Action and Step
        target = self.local_scale * local_planner + self.global_flag * self.global_scale * global_planner + self.state

        return target

    def draw_trajectory(self, save_name=None):
        plt.plot(self.path[0][:, 0], self.path[0][:, 1], 'mo', markersize=1, label="trajectory")  # add legend
        for i in range(1, len(self.path)):
            plt.plot(self.path[i][:, 0], self.path[i][:, 1], 'mo', markersize=1)
        if save_name: plt.savefig(save_name)

    def render(self):
        voronoi_plot_2d(self.cvt.vor, ax=ax, show_points=False, show_vertices=False, line_width=2)
        plt.plot(self.cvt.new_lines[0][0], self.cvt.new_lines[0][1], 'k', label='voronoi_ridges')  # add legend
        for line in self.cvt.new_lines:
            plt.plot(line[0], line[1], 'k')
        plt.imshow(self.cvt.distribution)
        plt.gca().invert_yaxis()
        plt.xlim(X_MIN - X_SIZE * plot_padding, X_MAX + X_SIZE * plot_padding)
        plt.ylim(Y_MIN - Y_SIZE * plot_padding, Y_MAX + Y_SIZE * plot_padding)
        plt.plot(self.cvt.vertices[:, 0], self.cvt.vertices[:, 1], 'bo', label='voronoi_vertices')
        plt.plot(self.cvt.centroids[:, 0], self.cvt.centroids[:, 1], 'go', label='voronoi_centroids')
        plt.plot(self.state[:, 0], self.state[:, 1], 'ro', label='robot_position')
        # # plt.plot(self.prev_state[:, 0], self.prev_state[:, 1], 'co', label='prev_robot_position')
        # # plt.quiver(self.state[:, 0], self.state[:, 1],
        # #            self.state[:, 0] - self.prev_state[:, 0],
        # #            self.state[:, 1] - self.prev_state[:, 1], angles='xy', scale_units='xy', scale=1)
        if show_path: self.draw_trajectory()
        plt.legend(loc="lower left")
        plt.title(f'number of robots = {self.robot_cnt}')
        if make_gif: video_maker(os.path.join(policy_dir, self.policy), fig_dir, self.timestep, done,
                                 video_type="mp4", dpi=200)
        plt.pause(0.1)
        plt.cla()

    def reset(self):
        self.cvt = CVT(self.distribution, self.robot_cnt, self.init_pos)
        self.prev_state = self.init_pos.copy()
        self.state = self.init_pos.copy()
        self.path = []
        self.cost = []
        self.policy = policy
        self.global_flag = True
        self.timestep = 0

    def step(self, action):
        self.prev_state = self.state.copy()
        self.path.append(self.state.copy())
        self.state += action
        self.state[:, 0] = self.state[:, 0].clip(X_MIN, X_MAX)
        self.state[:, 1] = self.state[:, 1].clip(Y_MIN, Y_MAX)
        self.timestep += 1
        if self.timestep % vor_duration == 0: self.cvt.step(self.state.copy())
        done = np.linalg.norm(action) < pos_error_thresh or self.timestep > max_timestep

        return done


if __name__ == '__main__':
    # Read Distribution/Density Map
    filename = './map/circle_distribution.npy'
    distribution = np.load(filename)
    robot_cnt = count_robot(distribution)
    scatter_ratio = np.random.random((robot_cnt, 2))
    robot_pos = interp([X_MIN, Y_MIN], [X_MAX, Y_MAX], scatter_ratio)  # generate robot on random initial positions

    # Args
    show_animation = True
    show_path = True
    make_gif = True

    # Save
    res_dir = "result"
    if os.path.exists(res_dir): shutil.rmtree(res_dir)
    os.mkdir(res_dir)
    cost_list = []

    for policy in ["local", "global", "exponential"]:
        # Initialize Environment
        env = RobotEnv(distribution, robot_pos, policy=policy)
        done = False

        # Render
        if show_animation:
            fig = plt.figure()
            ax = plt.gca()

        # Save
        policy_dir = os.path.join(res_dir, policy)
        os.mkdir(policy_dir)
        fig_dir = os.path.join(policy_dir, "fig")

        # Simulate
        while not done:
            if env.timestep % vor_duration == 0: target = env.get_target()
            action = env.get_action(target)
            done = env.step(action)
            if show_animation: env.render()
            print("Timestep: {}  Error: {:.4f}  Cost: {:.4f} Global: {}".format(env.timestep,
                                                                                np.linalg.norm(action),
                                                                                env.cost[-1],
                                                                                env.global_flag))
        plt.close()
        cost_list.append(np.insert(np.array(env.cost), 0, 10e7))
        np.save(os.path.join(policy_dir, "cost.npy"), cost_list[-1])

    # Plot Cost
    plt.plot(cost_list[0], label="local planner + standard map")
    plt.plot(cost_list[1], label="global planner + standard map")
    plt.plot(cost_list[2], label="local planner + exponential map")
    plt.title("Cost Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(res_dir, "cost.png"))
    plt.pause(1)
    plt.close()
