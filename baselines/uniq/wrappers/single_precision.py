"""Convert environment observations and spaces to single precision."""

import copy

import gym
import numpy as np
from gym.spaces import Box, Dict


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.observation_space, Box):
            space = self.observation_space
            self.observation_space = Box(space.low, space.high, space.shape)
        elif isinstance(self.observation_space, Dict):
            spaces = copy.copy(self.observation_space.spaces)
            for key, space in spaces.items():
                spaces[key] = Box(space.low, space.high, space.shape)
            self.observation_space = Dict(spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        if isinstance(observation, dict):
            observation = copy.copy(observation)
            for key, value in observation.items():
                observation[key] = value.astype(np.float32)
            return observation
        return observation
