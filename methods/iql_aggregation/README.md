# IQL-Agg

Gaussian-policy IQL with post-hoc SPA critic routing.

## Training

Run commands from `methods/iql_aggregation`.

All examples save checkpoints under `./checkpoints/<env_name>/` and TensorBoard logs under `./runs/<env_name>/`.

MuJoCo:

```bash
python train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/mujoco_config.py \
  --expectiles=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --save=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/ \
  --seed=2
```

AntMaze:

```bash
python train_offline.py \
  --env_name=antmaze-large-play-v2 \
  --config=configs/antmaze_config.py \
  --expectiles=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --eval_episodes=100 \
  --eval_interval=100000 \
  --save=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/ \
  --seed=2
```

Kitchen:

```bash
python train_offline.py \
  --env_name=kitchen-complete-v0 \
  --config=configs/kitchen_config.py \
  --expectiles=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --eval_episodes=10 \
  --save=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/ \
  --seed=2
```

`stagewise` runs Q/V training, frozen-critic SPA fitting, then SPA-routed AWR actor training. The critic and actor phases each default to `--max_steps=1000000`. The separate SPA phase defaults to `--spa_steps=200000`, including when refitting SPA from a checkpoint. `joint` is the appendix alternative.

The routing implementation is in `iql_aggregation/spa.py`.

## Domain configs

```text
configs/mujoco_config.py    expectile 0.7, AWR inverse temperature 3.0
configs/antmaze_config.py   expectile 0.9, AWR inverse temperature 10.0
configs/kitchen_config.py   expectile 0.7, AWR inverse temperature 0.5
```

The critic bank remains `{0.6, 0.7, 0.8, 0.9}` in all domains.

## Post-hoc loading

```bash
python train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/mujoco_config.py \
  --expectiles=0.6,0.7,0.8,0.9 \
  --load=True \
  --update_mar=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/post_hoc/ \
  --seed=2
```

The MuJoCo training example above creates the checkpoints used here. This command loads the critic bank and value representation, initializes fresh SPA heads, fits them, and trains a new actor. `--load_spa=True` and `--load_actor=True` are continuation options.

## Ablations

```text
--ablation_mode=0   Credited-AS routing (default)
--ablation_mode=1   AS-only routing
--ablation_mode=2   uniform aggregation
--ablation_mode=3   use the specified critic in the critic bank for every state
--ablation_mode=4   select the critic with the largest per-state Credited-AS
--ablation_mode=5   Shuffled Aggregation
```

`--single_critic_index` is zero-based. With the default critic bank `{0.6, 0.7, 0.8, 0.9}`, indices `0, 1, 2, 3` select the corresponding expectile critic. For example: `--ablation_mode=3 --single_critic_index=0`.

`--routing_alpha` is the routing temperature and `--credibility_beta` is the credibility scale; both default to `1.0`.

## Full reported sweep

The sweep completes all enabled tasks for each seed before moving to the next seed.

```bash
bash run_sweep.sh
```

This directory derives from the official IQL implementation: <https://github.com/ikostrikov/implicit_q_learning>.
