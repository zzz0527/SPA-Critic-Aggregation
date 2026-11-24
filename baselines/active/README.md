# ACTIVE Baseline

This directory contains the ACTIVE baseline used in the offline RL comparison experiments.

## Source

The implementation is adapted from the code released with the ACTIVE paper, *Offline Reinforcement Learning via Adaptive Imitation and In-Sample V-Ensemble*. The released source is available in the [ICLR 2025 supplemental archive](https://proceedings.iclr.cc/paper_files/paper/2025/file/c06f788963f0ce069f5b2dbf83fe7822-Supplemental-Conference.zip).

## Training

Run commands from `baselines/active`.

MuJoCo:

```bash
python active_iql_jax.py \
  --config_path configs/active_iql/mujoco_config.yaml \
  --gym_id hopper-medium-replay-v2 \
  --seed 2
```

AntMaze:

```bash
python active_iql_jax.py \
  --config_path configs/active_iql/antmaze_config.yaml \
  --gym_id antmaze-medium-play-v2 \
  --seed 2
```

Kitchen:

```bash
python active_iql_jax.py \
  --config_path configs/active_iql/kitchen_config.yaml \
  --gym_id kitchen-mixed-v0 \
  --seed 2
```

## Domain settings

| Domain | Expectile | AWR inverse temperature | Value ensemble | Evaluation episodes |
| --- | ---: | ---: | ---: | ---: |
| MuJoCo | 0.9 | 3.0 | 5 | 10 |
| AntMaze | 0.99 | 10.0 | 6 | 100 |
| Kitchen | 0.9 | 0.3 | 5 | 10 |

The provided `run_mujoco.sh`, `run_antmaze.sh`, and `run_kitchen.sh` scripts run all tasks in each domain with seeds `2, 4, 6, 8, 10`.

## Outputs

TensorBoard logs are written under the `log_dir` defined by the selected YAML. Set `--save True` to save model checkpoints in `<exp_name>__<gym_id>__seed<seed>__agent/` under the current directory.
