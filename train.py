import gymnasium as gym

from env_wrapper import register_env

import utils
from Models import PPO, DDQN
from reward_functions import RewardFunction, RewardFunctions

if __name__ == "__main__":
    utils.set_global_seed(0)

    hyperparameters_PPO = {
        "max_time_steps_per_batch": 8_000,
        "max_time_steps_per_episode": 2_000,
        "updates_per_iteration": 5,
        "save_freq": 10,
        "save_path": "SavedModels/PPO",
        "lr": 3e-3,
        "gamma": 0.99,
        "clip": 0.2
    }

    hyperparameters_DDQN = {
        'gamma': 0.99,
        'epsilon': 1.0,
        'lr': 5e-4,
        'batch_size': 64,
        'output_size': 4,
        'eps_end': 0.01,
        'input_size': [1],
        'max_mem_size':100000
        'eps_dec': 3e-4
        'tau': 0.01
        'update_target_every':100
        
    }

    register_env()
    reward_function = RewardFunction(RewardFunctions.KeepUpright)
    env = gym.make("RocketSim-v0", reward_function=reward_function)
    model = int(input("Choose model: 1=PPO, 2=DDQN"))
    if model ==1:
        agent = PPO(env, [16], hyperparameters=hyperparameters_PPO)
        agent.learn(100_000_000)
    elif model == 2:
        agent = DDQN(**hyperparameters_DDQN)
        # I still need to implement my learning function into my DDQN class as it is seperate currently - Jordan 28/09/2023
        
