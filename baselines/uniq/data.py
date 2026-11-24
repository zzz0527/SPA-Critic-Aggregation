"""D4RL loading and the disjoint UNIQ train/calibration split."""

from typing import NamedTuple, Tuple

import numpy as np


class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray


class ArrayDataset:
    def __init__(self, batch: Batch):
        lengths = {len(x) for x in batch}
        if len(lengths) != 1:
            raise ValueError("all dataset fields must have the same length")
        self.batch = Batch(*(np.asarray(x, dtype=np.float32) for x in batch))
        self.size = len(self.batch.rewards)

    def subset(self, indices: np.ndarray) -> "ArrayDataset":
        return ArrayDataset(Batch(*(x[indices] for x in self.batch)))

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Batch:
        indices = rng.randint(0, self.size, size=batch_size)
        return Batch(*(x[indices] for x in self.batch))


def split_train_calibration(
    dataset: ArrayDataset, calibration_fraction: float, seed: int
) -> Tuple[ArrayDataset, ArrayDataset]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(dataset.size)
    calibration_size = int(round(dataset.size * calibration_fraction))
    calibration_size = min(max(calibration_size, 1), dataset.size - 1)
    calibration_indices = permutation[:calibration_size]
    train_indices = permutation[calibration_size:]
    return dataset.subset(train_indices), dataset.subset(calibration_indices)


def _episode_returns(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    returns = []
    running = 0.0
    for reward, done in zip(rewards, dones):
        running += float(reward)
        if done:
            returns.append(running)
            running = 0.0
    if not returns or not dones[-1]:
        returns.append(running)
    return np.asarray(returns, dtype=np.float64)


def load_d4rl_dataset(env, env_name: str) -> ArrayDataset:
    # Importing d4rl can initialize MuJoCo, so keep it out of unit-test imports.
    import d4rl

    raw = d4rl.qlearning_dataset(env)
    actions = np.clip(raw["actions"], -1.0 + 1e-5, 1.0 - 1e-5)
    rewards = raw["rewards"].astype(np.float32).copy()
    terminals = raw["terminals"].astype(np.float32)
    masks = 1.0 - terminals

    timeouts = raw.get("timeouts", np.zeros_like(terminals)).astype(np.float32)
    episode_ends = np.zeros_like(terminals)
    for index in range(len(episode_ends) - 1):
        discontinuity = np.linalg.norm(
            raw["observations"][index + 1] - raw["next_observations"][index]
        ) > 1e-6
        episode_ends[index] = float(
            discontinuity or terminals[index] == 1.0 or timeouts[index] == 1.0
        )
    episode_ends[-1] = 1.0

    if "antmaze" in env_name:
        rewards -= 1.0
    elif any(name in env_name for name in ("halfcheetah", "hopper", "walker2d")):
        returns = _episode_returns(rewards, episode_ends)
        reward_range = float(np.max(returns) - np.min(returns))
        if reward_range <= 0.0:
            raise ValueError("cannot apply IQL reward normalization: zero return range")
        rewards = rewards / reward_range * 1000.0

    return ArrayDataset(
        Batch(
            observations=raw["observations"],
            actions=actions,
            rewards=rewards,
            masks=masks,
            next_observations=raw["next_observations"],
        )
    )
