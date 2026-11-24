# SPA Critic Aggregation

This code package contains the implementation of the manuscript **Post-Hoc Critic Comparison and Adaptive Routing via Split-Point Analysis for Offline Reinforcement Learning**.

This artifact contains the two method instantiations used in the paper (`IQL-Agg` and `IDQL-Agg`), along with comparison baselines and configurations.

## Package layout

```text
methods/
  iql_aggregation/       Gaussian-policy IQL-Agg.
  idql_aggregation/      Diffusion-policy IDQL-Agg.
baselines/
  active/                ACTIVE.
  idql/                  IDQL.
  iql/                   IQL.
  proj_iql/              Projective-IQL reproduction.
  uniq/                  UNIQ reproduction.
requirements.txt         Python and CUDA dependencies.
environment.yml          Conda environment for D4RL.
THIRD_PARTY_NOTICES.md    Upstream attribution.
```

The main method files are:

```text
methods/iql_aggregation/iql_aggregation/spa.py
methods/iql_aggregation/iql_aggregation/actor.py
methods/iql_aggregation/iql_aggregation/critic.py
methods/iql_aggregation/iql_aggregation/learner.py
methods/iql_aggregation/train_offline.py

methods/idql_aggregation/idql_aggregation/spa.py
methods/idql_aggregation/idql_aggregation/networks/state_value.py
methods/idql_aggregation/idql_aggregation/agent.py
methods/idql_aggregation/train_offline.py
```

## Paper-to-code mapping

| Paper quantity | Code |
| --- | --- |
| `MAR`, `MAR+`, `MAR-` (Eq. 6) | `mar`, `mar_up`, `mar_down` |
| `AS = MAR- / MAR` (Eq. 7) | `compute_spa_scores` |
| SDS (Eq. 10) | `compute_spa_scores` |
| Credited AS (Eq. 12) | `credited_as` |
| eligibility `AS > 1` and routing (Eq. 13) | `eligible_softmax_weights` |
| `Q_agg`, `V_agg` (Eq. 14) | weighted critic/value groups |
| IQL-Agg AWR extraction (Eq. 15) | `update_actor_aggregation` |
| IDQL-Agg candidate selection (Eq. 16) | `_eval_actions_impl` |

## Reproduction environment

The supported setup uses Conda on Linux or WSL2. Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate spa-critic-aggregation
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

MuJoCo 2.1 must be installed at `~/.mujoco/mujoco210`. Verify the installation with `python -c "import d4rl; print('D4RL import OK')"`.

The environment requires an NVIDIA driver compatible with CUDA 12 (version 525 or newer). Verify GPU access by running `python -c "import jax; print(jax.devices())"` and checking that the output contains a `GpuDevice`. IDQL-Agg always writes local evaluation files and also logs to W&B by default. Set `WANDB_MODE=offline` when network access is unavailable.

## Default protocol

If `training_strategy` is `stagewise`, IQL-Agg trains the expectile-indexed Q/V critic bank, fits SPA with Q/V frozen, and then trains the AWR actor with the critic frozen. IDQL-Agg trains the critic/value bank and diffusion actor together and then fits SPA with Q/V frozen. The critic and actor phases each default to `--max_steps=1000000`. The separate SPA phase defaults to 200,000 updates, with 1,000,000 updates for IDQL-Agg on AntMaze. `--spa_steps` overrides this default, which also applies when refitting SPA from a checkpoint.

If `training_strategy` is `joint`, the critic/value bank, actor, and SPA heads are updated together on every batch.

| Setting | Value |
| --- | --- |
| critic/value network | 2-layer MLP, width 256, ReLU |
| critic/value optimizer / learning rate | Adam / `3e-4` |
| critic and actor batch size | 256 |
| discount / critic target EMA | `0.99` / `0.005` |
| IDQL diffusion-actor optimizer / target EMA | AdamW / `0.001` |
| SPA module | one hidden layer per output head, width 64, ReLU |
| critic bank | `{0.6, 0.7, 0.8, 0.9}` |
| routing temperature `alpha` | `1.0` |
| credibility scale `beta` | `1.0` |
| numerical stabilizer | `1e-8` |
| reported seeds | `2, 4, 6, 8, 10` |

Domain-level IQL settings are:

| Domain | baseline expectile | AWR inverse temperature | evaluation episodes |
| --- | ---: | ---: | ---: |
| MuJoCo | 0.7 | 3.0 | 10 |
| AntMaze | 0.9 | 10.0 | 100 |
| Kitchen | 0.7 | 0.5 | 10 |

## Running IQL-Agg

The examples below write checkpoints to `./checkpoints/<env_name>/` and TensorBoard logs to `./runs/<env_name>/`.

MuJoCo:

```bash
cd methods/iql_aggregation
python3 train_offline.py \
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
python3 train_offline.py \
  --env_name=antmaze-medium-play-v2 \
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
python3 train_offline.py \
  --env_name=kitchen-mixed-v0 \
  --config=configs/kitchen_config.py \
  --expectiles=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --eval_episodes=10 \
  --save=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/ \
  --seed=2
```

`--save=True` saves the actor, critic bank, value networks, and SPA heads.

To fit SPA and a new actor from the saved critic bank:

```bash
python3 train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/mujoco_config.py \
  --expectiles=0.6,0.7,0.8,0.9 \
  --load=True \
  --update_mar=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/post_hoc/ \
  --seed=2
```

By default, `--load` restores Q together with the value trunk and expectile head. The SPA heads and actor are initialized afresh. Use `--load_spa=True` or `--load_actor=True` to continue training from their saved parameters.

IQL-Agg ablations use `--ablation_mode`:

- `0` uses Credited-AS routing (the paper default).
- `1` uses AS-only routing.
- `2` uses uniform aggregation.
- `3` selects the same critic for every state, with its index in the critic bank specified by `--single_critic_index`.
- `4` selects the critic with the largest Credited-AS for each state.
- `5` uses Shuffled Aggregation.

## Running IDQL-Agg

```bash
cd methods/idql_aggregation
WANDB_MODE=offline python3 train_offline.py \
  --env_name=hopper-medium-replay-v2 \
  --config=configs/idql_config.py \
  --value_taus=0.6,0.7,0.8,0.9 \
  --training_strategy=stagewise \
  --routing_temperature=1.0 \
  --credibility_scale=1.0 \
  --save=True \
  --base_dir=./checkpoints/ \
  --save_dir=./runs/ \
  --seed=2
```

`--inference_Ns` controls the candidate counts evaluated at test time. `--N` controls the training-time diffusion setting. To fit fresh SPA heads without repeating diffusion actor and critic/value training, run the same command with `--load=True --update_mar=True` and the matching `--base_dir` and seed. Use `--load_spa=True --update_mar=False` to restore and evaluate the complete saved model without further training.

## Full reported sweep

Each method directory contains one sweep script. Each script completes all enabled tasks for a seed before moving to the next seed.

```bash
bash methods/iql_aggregation/run_sweep.sh
bash methods/idql_aggregation/run_sweep.sh
```

## Outputs

IQL-Agg writes TensorBoard logs and score files under `<save_dir>/<env_name>/`. IDQL-Agg writes W&B logs to the directory specified by `--wandb_dir`. Both methods write evaluation returns to `.csv` files for quick inspection.

IQL-Agg checkpoints use:

```text
<base_dir>/<env_name>/
  actor.ckpt
  critics/critic_<tau>_seed<seed>.ckpt
  values/value_<tau>_seed<seed>.ckpt
```

IDQL-Agg stores the diffusion actor, actor target, Q bank, target Q bank, value representation, SPA heads, and optimizer states in a checkpoint for each seed.

```text
<base_dir>/<env_name>/seed_<seed>.ckpt
```

## Baselines and attribution

Each baseline directory contains its source note and domain examples. IQL-Agg is based on the public JAX IQL implementation, and IDQL-Agg is based on IDQL. The Projective-IQL and UNIQ reproductions extend the repository's IQL baseline because no official implementations were used. See `THIRD_PARTY_NOTICES.md` for path-level attribution.
