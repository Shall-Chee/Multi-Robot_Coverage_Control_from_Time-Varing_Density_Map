import torch
import numpy as np
import pygame
import gym
import random
import argparse

class Goal2Goal:
    def __init__(self, arglist):
        # Agents initialization:
        self.num_agents = arglist.num_agents # default number is 6
        self.max_steps = arglist.max_steps # default maximum episode length is 200
        self.lr = arglist.lr # learning rate
        self.gamma = arglist.gamma # default discount factor is 0.95
        self.batch_size = arglist.batch_size # default batch size is 1024
        # self.obs_vec = None
        self.step_func_name = arglist.step_name
        self.step_func = self.get_step

        '''
        goal positions: 1 agent for 1 goal
        you can also set goal positions in args (goals input dimension: N * 2, N: number of goals)
        '''
        self.goals = self.get_goals

        '''
        Human Social Force model parameters:
        '''
        self.eps = 1e-5
        self.v0 = 1.34
        self.vh_max = 1.3 * self.v0
        self.V0 = 10.0
        self.U0 = 10.0
        self.sigma = 2.0
        self.R = 2.0
        self.tau = 0.5
        self.dt = 0.1
        self.hdt = 2.0
        self.phi = 10.0 * np.pi / 9.0
        self.c = 0.5

        '''
        pygame rendering parameters:
        '''
        self.screen_size = [1600, 800]
        self.background_color = [255, 255, 255]
        self.wall_color = [0, 0, 0]
        self.agent_color = [0, 0, 255]
        self.goal_color = [255, 0, 0]
        self.agent_radius = 9
        self.goal_radius = 4
        self.width = 3

        self.pygame_init = False

    def get_parmas(self, args):
        # get parameters from arguments:
        self.num_agents = args.getint('num_agents')
        self.num_obs = args.getint('num_obs')
        self.num_actions = args.getint('num_actions')
        self.goals = args.getlist('goal_positions')

    @property
    def get_goals(self):
        goals = [[50.0 + random.uniform(-20.0, 2.0), 30.0 + random.uniform(-20.0, 20.0)] for _ in range(self.num_agents)]
        return np.array(goals)

    '''
    step function for RL training
    input: N * 2 actions array
    return: N * 4 agent states, float reward, bool done, N * 2 observations
    '''
    def step(self, action):
        N = self.num_agents
        self.steps += 1
        pos, velo = self.decode_state(self.state_agents)
        a = action

        # get observations as a N * 2 vector array:
        self.obs_vec = self.goals - pos
        e = self._norm(self.obs_vec)
        pos_new, velo_new = np.copy(pos), np.copy(velo)

        # update states:
        velo_new = a
        pos_new += velo_new * self.dt

        # encode states:
        self.state_agents = self.encode_state(pos_new, velo_new)

        # compute reward:
        reward = 0.0
        for k in range(N):
            reward -= np.sum(np.square(pos_new[k] - self.goals[k]))

        done = self.steps >= self.max_steps
        return self.state_agents, reward, done, self.obs_vec

    '''
    simple controller for testing:
    input: N * 2 actions array
    return: N * 4 agent states, float reward, bool done, N * 2 observations
    '''
    def vanilla(self, action):
        N = self.num_agents
        pos, velo = self.decode_state(self.state_agents)

        # get observations as a N * 2 numpy array:
        self.obs_vec = self.goals - pos
        e = self._norm(self.obs_vec)
        velo_new = 0.2 * self.obs_vec

        # constrain velocity changes:
        velo_new_clip = np.clip(velo_new, -30.0, 30.0)
        self.velo_new = velo_new_clip
        pos_new = pos + velo_new_clip * self.dt

        # encode states:
        self.state_agents = self.encode_state(pos_new, velo_new)

        # compute reward:
        reward = 0.0
        for k in range(N):
            reward -= np.sum(np.square(pos_new[k] - self.goals[k]))

        done = self.steps >= self.max_steps

        return self.state_agents, reward, done, self.obs_vec

    '''
    step function using Human Social Force model as controller:
    input: N * 2 actions array
    return: N * 4 agent states, float reward, bool done, N * 2 observations
    '''
    def hsf(self, action):
        N = self.num_agents
        pos, velo = self.decode_state(self.state_agents)
        self.obs_vec = self.goals - pos
        e = self._norm(self.obs_vec)

        pos_new, velo_new = np.copy(pos), np.copy(velo)

        # Compute human social force:
        for i in range(N):
            # force to goal:
            F = 5 * (self.v0 * e[i] - velo[i]) / self.tau

            # force from other agents:
            for j in range(N):
                r0 = pos[i] - pos[j]
                s = np.linalg.norm(velo[j]) * self.hdt
                n0 = np.linalg.norm(r0)
                n1 = np.linalg.norm(r0 - s * e[j])
                b = 0.5 * np.sqrt(np.max([np.square(n0+n1) - np.square(s), 0.0]))
                f = 0.25 * self.V0 * np.exp(-b / self.sigma) * (2.0 + n0 / (n1 + self.eps) + n1 / (n0 + self.eps)) * r0 / (self.sigma * b + self.eps)
                w = 1.0 if np.dot(e[j], -f) >= np.linalg.norm(f) * np.cos(self.phi) else self.c
                F += w * f

            # update velocities:
            w_all = velo[i] + F * self.dt
            velo_new[i] = self._norm(w_all) * np.min([np.linalg.norm(w_all), self.vh_max])

            # update agents positions:
            pos_new[i] = pos[i] + velo_new[i] * self.dt

        # encode states:
        self.state_agents = self.encode_state(pos_new, velo_new)

        # compute reward:
        reward = 0.0
        for k in range(N):
            reward -= np.sum(np.square(pos_new[k] - self.goals[k]))

        done = self.steps >= self.max_steps

        return self.state_agents, reward, done, self.obs_vec

    def reset(self):
        N = self.num_agents
        # set initial positions for agents:
        pos_agents = np.concatenate([np.array([100.0, 20.0])
                                     + 2.0 * (2.0 * np.random.uniform(size=2) )- 1 for _ in range(N)])
        self.pos_agents = pos_agents.reshape((N,2))

        # set initial velocities for agents:
        velo_agents = np.zeros(shape=2*N)
        self.velo_agents = velo_agents.reshape((N,2))
        self.state_agents = np.concatenate((self.pos_agents, self.velo_agents), axis=1)
        self.steps = 0

        return self.state_agents

    def decode_state(self, state):
        # extract agents positions and velocities as two N * 2 arrays:
        pos = state[:,:2]
        velo = state[:,2:]
        return pos, velo

    def encode_state(self, pos, velo):
        return np.concatenate((pos, velo), axis=1)

    def _norm(self, x):
        return x / (np.linalg.norm(x) + self.eps)

    def get_obs(self):
        pos, goal = self.pos_agents, self.goals
        obs = goal - pos
        return obs

    def render(self):
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            pass

        pos, velo = self.decode_state(self.state_agents)
        N = self.num_agents

        # start rendering:
        self.screen.fill(self.background_color)

        for i in range(N):
            position = (10.0 * pos[i]).astype(int).tolist()
            end_position = (10.0 * (pos[i] + self._norm(velo[i]))).astype(int).tolist()
            goal_position = (10.0 * self.goals[i]).astype(int).tolist()

            # goal:
            pygame.draw.circle(self.screen, self.goal_color, goal_position, self.goal_radius, self.width)

            # agents:
            pygame.draw.circle(self.screen, self.agent_color, position, self.agent_radius, self.width)
            pygame.draw.line(self.screen, self.goal_color, position, end_position, self.width)

        pygame.display.flip()
        self.clock.tick(30)

    @property
    def get_step(self):
        if self.step_func_name == "vanilla":
            print("Now using vanilla!")
            return self.vanilla
        elif self.step_func_name == "rl":
            print("Now start training!")
            return self.step
        else:
            print("Now using hsf model!")
            return self.hsf

def parse_args():
    parser = argparse.ArgumentParser("Enter step function name, choices: vanilla, hsf")
    parser.add_argument("--step-name", type=str, default="vanilla", help="name of step function")
    parser.add_argument("--max-steps", type=int, default=200, help="maximum episode length")
    parser.add_argument("--num-agents", type=int, default=6, help="number of agents")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    return parser.parse_args()


def main():
    arg_list = parse_args()
    env = Goal2Goal(arg_list)
    state = env.reset()
    N = env.num_agents
    test_action = np.ones(2*N).reshape((N,2))
    # print(test_action)
    while True:
        env.render()
        next_state, reward, done, obs = env.step_func(test_action)


if __name__ == '__main__':
    main()





