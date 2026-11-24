from typing import Dict

import gym
import numpy as np

from idql_aggregation.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    ensemble_indices = []
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(observation)
            if hasattr(agent, "last_ensemble_idx"):
                ensemble_indices.append(float(np.asarray(agent.last_ensemble_idx)))
            observation, _, done, _ = env.step(action)
    info = {"return": np.mean(env.return_queue)}
    if len(ensemble_indices) > 0:
        info["ensemble_idx_mean"] = float(np.mean(ensemble_indices))
        info["ensemble_idx_last"] = float(ensemble_indices[-1])
    return info

def implicit_evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.sample_implicit_policy(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}
