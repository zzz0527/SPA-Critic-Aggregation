# IDQL Baseline

This directory contains the DDPM-IDQL baseline used for comparison with IDQL-Agg.

## Source

The implementation is adapted from the [official IDQL repository](https://github.com/philippe-eecs/IDQL) for the paper [IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/abs/2304.10573).

## Training

Run commands from `baselines/idql`. Every run writes local evaluation score files and also logs to W&B by default. Set `WANDB_MODE=offline` when network access is unavailable.

MuJoCo:

```bash
WANDB_MODE=offline python train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/idql_config.py \
  --save_dir=./runs/ \
  --wandb_dir=./wandb/ \
  --seed=2
```

AntMaze:

```bash
WANDB_MODE=offline python train_offline.py \
  --env_name=antmaze-medium-play-v2 \
  --config=configs/idql_config.py \
  --save_dir=./runs/ \
  --wandb_dir=./wandb/ \
  --seed=2
```

Kitchen:

```bash
WANDB_MODE=offline python train_offline.py \
  --env_name=kitchen-mixed-v0 \
  --config=configs/idql_config.py \
  --save_dir=./runs/ \
  --wandb_dir=./wandb/ \
  --seed=2
```

## Domain settings

The baseline uses the same network configuration in all domains. The critic expectile is 0.9 for AntMaze and 0.7 otherwise. Evaluation defaults to 100 episodes for AntMaze and 10 episodes for MuJoCo and Kitchen.

Each sampled batch synchronously updates the diffusion actor, value network, and Q-functions through `agent.update`. Training uses one million updates with batch size 256, and `--inference_Ns=16,64,256` supplies the default evaluation candidate counts.

## Outputs

For each candidate count in `--inference_Ns`, normalized evaluation returns are written to `<save_dir>/<env_name>/<seed>_N<N>.txt`. Each row contains `step`, normalized return, the latest-three mean, and the latest-three population standard deviation. Online fine-tuning uses `<seed>_online_N<N>.txt`.

W&B data is written under `--wandb_dir`. Use `WANDB_MODE=online` for cloud logging, `WANDB_MODE=offline` for local W&B logging, or `--use_wandb=False` to disable W&B. The reported seeds are `2, 4, 6, 8, 10`.
