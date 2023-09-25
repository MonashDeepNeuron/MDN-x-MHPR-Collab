import gymnasium as gym

from env_wrapper import register_env
from reward_functions import RewardFunction, RewardFunctions

register_env()
env = gym.make("RocketSim-v0", reward_function=RewardFunction(RewardFunctions.TestReward))

observation, _ = env.reset()
print(observation)

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, _ = env.step(0)
    print(observation["altitude"], end="\r")

    if terminated or truncated:
        print()
        print("----------------- Env Reset -----------------")
        observation, info = env.reset()

