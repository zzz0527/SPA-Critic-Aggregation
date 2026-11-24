# IDQL-Agg

Diffusion-policy IDQL with post-hoc SPA critic routing. Diffusion proposes candidate in-distribution actions, and SPA selects the action with the largest aggregated Q-value.

## Training

Run commands from `methods/idql_aggregation`.

All examples write local evaluation score files under `./runs/<env_name>/`, save seed-specific checkpoints under `./checkpoints/<env_name>/`, and also log to W&B. Set `WANDB_MODE=offline` when network access is unavailable.

MuJoCo:

```bash
WANDB_MODE=offline python3 train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/idql_config.py \
  --value_taus=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --routing_temperature=1.0 \
  --credibility_scale=1.0 \
  --eval_episodes=10 \
  --save=True \
  --save_dir=./runs/ \
  --base_dir=./checkpoints/ \
  --seed=2
```

AntMaze:

```bash
WANDB_MODE=offline python3 train_offline.py \
  --env_name=antmaze-large-play-v2 \
  --config=configs/idql_config.py \
  --value_taus=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --routing_temperature=1.0 \
  --credibility_scale=1.0 \
  --eval_episodes=100 \
  --eval_interval=100000 \
  --save=True \
  --save_dir=./runs/ \
  --base_dir=./checkpoints/ \
  --seed=2
```

Kitchen:

```bash
WANDB_MODE=offline python3 train_offline.py \
  --env_name=kitchen-complete-v0 \
  --config=configs/idql_config.py \
  --value_taus=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --routing_temperature=1.0 \
  --credibility_scale=1.0 \
  --eval_episodes=10 \
  --save=True \
  --save_dir=./runs/ \
  --base_dir=./checkpoints/ \
  --seed=2
```

`stagewise` trains the expectile value bank, grouped-Q critics, and diffusion actor together for `--max_steps=1000000` updates by default. It then freezes the critic/value bank and fits SPA for 1,000,000 updates on AntMaze and 200,000 otherwise. `--spa_steps` overrides this default. The same SPA budget applies when refitting it from a checkpoint. The actor loss is independent of the critic. `joint` updates the actor, critic/value bank, and SPA heads together on every batch.

The routing implementation is in `idql_aggregation/spa.py`.

## Post-hoc loading

```bash
WANDB_MODE=offline python3 train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/idql_config.py \
  --value_taus=0.6,0.7,0.8,0.9 \
  --load=True \
  --update_mar=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/post_hoc/ \
  --seed=2
```

This loads the seed-specific diffusion actor, grouped-Q critics, target critics, and expectile value representation, initializes fresh SPA heads, and fits SPA without repeating actor/Q/V training. Use `--load_spa=True --update_mar=False` to restore and evaluate the already fitted SPA heads without further training.

## Evaluation

Under `stagewise`, the critic/value bank and diffusion actor are trained first without evaluation because SPA is not fitted yet. Evaluation starts during the subsequent SPA phase at every `--eval_interval`, and one final evaluation is guaranteed after SPA finishes. Under `joint`, evaluation runs every `--eval_interval` and once at the end if needed.

At each evaluation point, IDQL-Agg separately evaluates every candidate count in `--inference_Ns` (default: `16,64,256`). For each state, the target diffusion policy samples `N` candidate actions. SPA computes state-dependent Credited-AS routing weights in the critic bank, the scores are aggregated for every candidate, and the candidate with the largest aggregated Q-value is executed.

The policy is rolled out for 100 episodes on AntMaze and 10 episodes otherwise. The mean raw return is converted to a D4RL-normalized score. Each `N` is recorded in a separate file:

```text
<save_dir>/<env_name>/<seed>_N<N>.csv
```

Each CSV includes a header row. Each data row contains `step`, normalized return, the mean of the latest three evaluations, and their population standard deviation, matching the four-column IQL-Agg score format. Online fine-tuning, when enabled, uses `<seed>_online_N<N>.csv`.

W&B logging is enabled by default, while the local `.csv` files are always written. Use `WANDB_MODE=online` for cloud logging, `WANDB_MODE=offline` for local W&B logging, or `--use_wandb=False` to disable W&B entirely. `--wandb_dir` selects the W&B data directory.

## Domain configs

```text
configs/idql_config.py    shared diffusion-policy and network configuration
```

The critic bank remains `{0.6, 0.7, 0.8, 0.9}` in all domains.

## Full reported sweep

The sweep completes all 18 tasks for each seed before moving to the next seed.

```bash
bash run_sweep.sh
```

The launcher writes evaluation files under `./runs/` and uses offline W&B logging by default. Set `WANDB_MODE=online` to upload runs directly.

This directory derives from the official IDQL implementation: <https://github.com/philippe-eecs/IDQL>.
