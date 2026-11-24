# IQL Baseline

This directory contains the plain IQL baseline used for comparison with IQL-Agg and as the codebase for the Projective-IQL and UNIQ reproductions.

## Source

The implementation is adapted from the [official IQL repository](https://github.com/ikostrikov/implicit_q_learning) for the paper [Offline Reinforcement Learning with Implicit Q-Learning](https://openreview.net/forum?id=68n2s9ZJWF8).

## Training

Run commands from `baselines/iql`.

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

Training uses one million updates with batch size 256. The reported seeds are `2, 4, 6, 8, 10`.

## Outputs

Evaluation scores are written to `<save_dir>/<env_name>/<seed>.txt`. Each row contains `step`, normalized return, the latest-three mean, and the latest-three population standard deviation. TensorBoard logs are written under `<save_dir>/<env_name>/tb/<seed>/`.
