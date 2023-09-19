import gymnasium as gym
import env_wrapper


env = gym.make("rocket_sim-v0")

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

