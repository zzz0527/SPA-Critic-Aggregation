# Projective-IQL Baseline

This directory contains the Projective-IQL reproduction used in the comparison experiments. It is implemented by extending the IQL baseline in `baselines/iql`; no official Projective-IQL code release was used.

## Source

The algorithm follows the paper [Projective Implicit Q-Learning with Support Constraint for Offline Reinforcement Learning](https://arxiv.org/abs/2501.08907), while the training entry point, D4RL processing, evaluation, networks, and domain configurations are based on the repository's IQL baseline.

## Training

Run commands from `baselines/proj_iql`.

MuJoCo:

```bash
python train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/mujoco_config.py \
  --save_dir=./runs/ \
  --seed=2
```

AntMaze:

```bash
python train_offline.py \
  --env_name=antmaze-medium-play-v2 \
  --config=configs/antmaze_config.py \
  --eval_episodes=100 \
  --eval_interval=100000 \
  --save_dir=./runs/ \
  --seed=2
```

Kitchen:

```bash
python train_offline.py \
  --env_name=kitchen-mixed-v0 \
  --config=configs/kitchen_config.py \
  --save_dir=./runs/ \
  --seed=2
```

## Domain settings

| Domain | Config | Expectile | AWR inverse temperature | Evaluation episodes |
| --- | --- | ---: | ---: | ---: |
| MuJoCo | `configs/mujoco_config.py` | 0.7 | 3.0 | 10 |
| AntMaze | `configs/antmaze_config.py` | 0.9 | 10.0 | 100 |
| Kitchen | `configs/kitchen_config.py` | 0.7 | 0.5 | 10 |

These shared settings match the IQL baseline. Projective-IQL additionally pretrains a behavior policy for 100,000 updates by default and reuses its checkpoint when available. The reported seeds are `2, 4, 6, 8, 10`.

## Outputs

Evaluation scores are written to `<save_dir>/<env_name>/<seed>.txt`. Each row contains `step`, normalized return, the latest-three mean, and the latest-three population standard deviation. TensorBoard logs and behavior-policy checkpoints are also written under `<save_dir>/<env_name>/`.
