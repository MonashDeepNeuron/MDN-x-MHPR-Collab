import gymnasium as gym

from env_wrapper import register_env
from reward_functions import RewardFunction, RewardFunctions


register_env()
env = gym.make("RocketSim-v0", reward_function=RewardFunction(RewardFunctions.TestReward))

observation, _ = env.reset()
print(observation)

peak_alt = observation["altitude"]
print(peak_alt)

while True:
    action = env.action_space.sample()


    observation, reward, terminated, truncated, _ = env.step(action)
    peak_alt = max(peak_alt, observation["altitude"])
    print(observation["altitude"], peak_alt)

    if terminated or truncated:
        print()
        print("----------------- Env Reset -----------------")
        observation, info = env.reset()

