import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_wrapper import register_env

import utils
from Models import PPO
from reward_functions import RewardFunction, RewardFunctions

if __name__ == "__main__":
    utils.set_global_seed(0)

    hyperparameters = {
        'time_steps_per_batch': 8000,
        'max_time_steps_per_episode': 100,
        'gamma': 0.99,
        'updates_per_iteration': 10,
        'lr': 1e-4,
        'clip': 0.2,
    }

    register_env()
    reward_function = RewardFunction(RewardFunctions.KeepUpright)
    env = gym.make("RocketSim-v0", reward_function=reward_function)

    agent = PPO(env, [16], hyperparameters=hyperparameters)

    agent.learn(100_000_000)
