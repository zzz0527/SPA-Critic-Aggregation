"""Deterministic policy evaluation shared with the IQL baselines."""

from typing import Dict

import gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {"return": [], "length": []}
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for key in stats:
            stats[key].append(info["episode"][key])
    return {key: float(np.mean(values)) for key, values in stats.items()}
