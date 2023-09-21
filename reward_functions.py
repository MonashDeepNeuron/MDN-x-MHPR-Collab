from enum import Enum

import numpy as np


class RewardFunctions(Enum):
    KeepUpright = "keep_upright"
    TestReward = "test_reward"


class RewardFunction:
    def __init__(self, func: RewardFunctions, env):
        self.env = env
        self.function = None
        self.function_enum = func
        self._set_function()

    def _set_function(self):
        self.function = globals()[self.function_enum.value]

    def set_function(self, func: RewardFunctions):
        self.function_enum = func
        self._set_function()

    def get_reward(self):
        return self.function(self.env)


# Define reward functions down here.
def keep_upright(env):
    euler_angles = np.rad2deg(env.sim.state.euler_angles)

    initial_angle = np.array([0, 90.0, 0])

    sum_abs_euler_angles = -np.linalg.norm(euler_angles - initial_angle)

    if sum_abs_euler_angles >= -0.5:
        return 1
    else:
        return sum_abs_euler_angles


# Testing
def test_reward(env: None):
    return 7


if __name__ == "__main__":
    reward_func = RewardFunction(RewardFunctions.TestReward, None)
    print("Probably working fine" if reward_func.get_reward() == test_reward(None) else "Something went wrong")

