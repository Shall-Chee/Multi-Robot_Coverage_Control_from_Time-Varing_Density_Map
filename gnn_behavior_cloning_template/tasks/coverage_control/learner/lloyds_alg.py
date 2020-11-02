import numpy as np


def execute_lloyds(observation):
    rel_centroids, _, _ = observation
    # Lloyd's algorithm: Drive to the centroid
    vel = 4.0 * rel_centroids
    return vel


def train_lloyds(env, args, device):
    debug = args.getboolean('debug')

    rewards = []
    total_numsteps = 0
    updates = 0

    n_train_episodes = args.getint('n_train_episodes')

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    for i in range(n_train_episodes):

        state = env.reset()

        done = False
        policy_loss_sum = 0
        n_steps = 0
        while not done and n_steps <= args.getint('max_episode_length'):
            optimal_action = execute_lloyds(state)
            next_state, reward, done, _ = env.step(optimal_action)
            if args.getboolean("render"):
                env.render()

            state = next_state
            n_steps += 1
            mean_reward = reward  # TODO
            if debug:
                print(
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {}, policy loss: {}".format(
                        i, updates,
                        total_numsteps,
                        mean_reward,
                        policy_loss_sum))

    env.close()
    return stats
