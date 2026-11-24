#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python_bin=${PYTHON_BIN:-python3}
"$python_bin" -c 'import jax; devices = jax.devices(); assert any(d.platform == "gpu" for d in devices), f"No JAX GPU device found: {devices}"'

tasks=(
  halfcheetah-medium-v2
  hopper-medium-v2
  walker2d-medium-v2
  halfcheetah-medium-replay-v2
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
  halfcheetah-medium-expert-v2
  hopper-medium-expert-v2
  walker2d-medium-expert-v2
  antmaze-umaze-v2
  antmaze-umaze-diverse-v2
  antmaze-medium-play-v2
  antmaze-medium-diverse-v2
  antmaze-large-play-v2
  antmaze-large-diverse-v2
  kitchen-complete-v0
  kitchen-partial-v0
  kitchen-mixed-v0
)

seeds=(2 4 6 8 10)

for seed in "${seeds[@]}"; do
  for env_name in "${tasks[@]}"; do
    config=mujoco_config.py
    eval_interval=5000
    if [[ "$env_name" == antmaze-* ]]; then
      config=antmaze_config.py
      eval_interval=100000
    elif [[ "$env_name" == kitchen-* ]]; then
      config=kitchen_config.py
    fi

    echo "Running env=$env_name seed=$seed"
    "$python_bin" train_offline.py \
      --env_name="$env_name" \
      --config="configs/$config" \
      --expectiles=0.6,0.7,0.8,0.9 \
      --training_strategy=stagewise \
      --routing_alpha=1.0 \
      --credibility_beta=1.0 \
      --max_steps=1000000 \
      --spa_steps=200000 \
      --batch_size=256 \
      --eval_episodes=0 \
      --eval_interval="$eval_interval" \
      --seed="$seed" \
      --save=True \
      --save_dir=./runs/ \
      --base_dir=./checkpoints/
  done
done
