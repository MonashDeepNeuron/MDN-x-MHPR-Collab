from enum import Enum

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register

import Modules.MonteCarloTools as MCTools
from Modules.run_monte_carlo_sweep import MonteCarlo
from Modules.Structures import CoordinateSystemType, FrameType
# from Modules.Sensors import TVCAction
from Modules.main import Simulation

import gc

import numpy as np

import utils
from reward_functions import RewardFunction


class ActionSpace(Enum):
    DO_NOTHING = 0
    TILT_UP = 1
    TILT_DOWN = 2
    PAN_UP = 3
    PAN_DOWN = 4


class RocketSim(gym.Env):
    def __init__(self, reward_function: RewardFunction):
        # TODO: Update these to be scaled nicely.
        self.observation_space = spaces.Dict({
            "altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            "displacement": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "velocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "acceleration": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "euler_angles": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "angular_velocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64)
        })

        self.action_space = spaces.Discrete(5)

        self.reward_function = reward_function

        self.monte_carlo = MonteCarlo("mdn_mcConfig.yaml", "mdn_vehicleConfig.yaml")
        self.monte_carlo_config = None
        self.sim = None
        self.sim_count = 0
        self.initial_alt = None
        self.initial_pos = None
        self.transform_matrix = None

    def _get_obs(self):
        altitude = self.sim.state.getAltitudeGeometric() - self.initial_alt

        # Position shouldn't be returned as it is based on the initial starting location which is config specific
        position = self.sim.state.getPosition(CoordinateSystemType.GEOCENTRIC)

        displacement = self.transform_matrix @ (position - self.initial_pos)
        velocity = self.sim.state.getVelocity(coord=CoordinateSystemType.BODY, frame=FrameType.EARTH)
        acceleration = self.sim.state.getAcceleration(coord=CoordinateSystemType.BODY, frame=FrameType.EARTH)
        euler_angles = self.sim.state.euler_angles
        angular_velocity = self.sim.state.getAngularVelocity(coord=CoordinateSystemType.BODY, frame=FrameType.EARTH)

        return {
            "altitude": np.array([altitude]),  # to make it match the obs_shape
            "displacement": transpose(displacement),
            "velocity": transpose(velocity),
            "acceleration": transpose(acceleration),
            "euler_angles": euler_angles,
            "angular_velocity": transpose(angular_velocity)
        }

    def has_reached_burnout(self):
        return (np.abs(self.sim.propulsion.getForce(self.sim.state)[0][0]) == 0) and (self.sim.state.time > self.sim.state.dt)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Perform memory cleanup if this isn't the first run
        # I don't know if this is important as we replace sim each time but cant hurt
        if self.sim is not None:
            # Keeping memory usage in check
            del self.sim.aerodynamics.force_coefficients
            del self.sim.aerodynamics.moment_coefficients
            try:
                del self.sim.atmosphere.pyatm
            except AttributeError:
                pass

            gc.collect()  # Runs the garbage collection.

        utils.block_print()
        _, self.monte_carlo_config, _ = self.monte_carlo._generate_cases()[0]

        self.sim = self.monte_carlo._initialise_simulation("RocketSim", self.monte_carlo_config, self.sim_count)
        utils.enable_print()
        self.sim_count += 1

        self.initial_alt = self.sim.state.getAltitudeGeometric()
        self.initial_pos = self.sim.state.getPosition(CoordinateSystemType.GEOCENTRIC)
        self.transform_matrix = self.sim.state.getTransformationMatrix(CoordinateSystemType.GEOGRAPHIC,
                                                                       CoordinateSystemType.GEOCENTRIC)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        mapped_action = ActionSpace(action)

        # Not using their TVCAction enum as it does not include the do nothing action.
        if mapped_action == ActionSpace.TILT_UP:
            self.sim.tilt += self.sim.tilt_stepsize
        elif mapped_action == ActionSpace.TILT_DOWN:
            self.sim.tilt -= self.sim.tilt_stepsize
        elif mapped_action == ActionSpace.PAN_UP:
            self.sim.pan += self.sim.pan_stepsize
        elif mapped_action == ActionSpace.PAN_DOWN:
            self.sim.pan -= self.sim.pan_stepsize
        elif mapped_action == ActionSpace.DO_NOTHING:
            pass
        else:
            raise ValueError(f"Action {action} not recognised.")

        # Enforcing a hard limit on tilt and pan
        self.sim.tilt = max(-self.sim.max_tilt, min(self.sim.max_tilt, self.sim.tilt))
        self.sim.pan = max(-self.sim.max_pan, min(self.sim.max_pan, self.sim.pan))

        utils.block_print()
        self.sim.iteration()
        utils.enable_print()

        obs = self._get_obs()
        reward = self.reward_function.get_reward(self)
        terminated = self.has_reached_burnout()  # Using this for if sim is over, can be changed.
        truncated = False  # We prob not need this
        return obs, reward, terminated, truncated, {}


def register_env():
    register(id="RocketSim-v0", entry_point="env_wrapper:RocketSim")


# Utility
def transpose(lst):
    return np.array(list(map(list, zip(*lst)))[0])
