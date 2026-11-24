"""Episode statistics and D4RL score normalization wrapper."""

import gym
import numpy as np

from wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            episode_return = self.reward_sum
            if hasattr(self, "get_normalized_score"):
                episode_return = self.get_normalized_score(episode_return) * 100.0
            info["episode"] = {
                "return": episode_return,
                "length": self.episode_length,
            }
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
