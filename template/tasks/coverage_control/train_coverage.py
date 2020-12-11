from os import path
import argparse
import configparser
import numpy as np
import random
import gym
import torch

import envs
from learner.lloyds_alg import train_lloyds
from learner.gnn_cloning import train_cloning


def run_experiment(args, test=False):
    if test:
        print("Running train_coverage.py in testing mode...")
        args['n_episodes'] = '1'
        args['render'] = 'False'
        args['max_episode_length'] = '2'
    
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)

    env.env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alg = args.get('alg').lower()

    if alg == 'lloyds':
        stats = train_lloyds(env, args, device)
    elif alg == 'cloning':
        stats = train_cloning(env, args, device)
    else:
        raise Exception('Invalid algorithm/mode name')

    return stats


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('fname', type=str,
                        help='Name of configuration .cfg file')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run in testing mode for code purposes ')

    args = parser.parse_args()
    fname = args.fname
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            stats = run_experiment(config[section_name], test=args.test)
            print(section_name + ", " + str(stats['mean']) + ", " + str(stats['std']))
    else:
        val = run_experiment(config[config.default_section])
        print(val)


if __name__ == "__main__":
    main()
