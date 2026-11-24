CONFIG=configs/active_iql/mujoco_config.yaml
ENVS=(
  halfcheetah-medium-v2
  hopper-medium-v2
  walker2d-medium-v2
  halfcheetah-medium-replay-v2
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
  halfcheetah-medium-expert-v2
  hopper-medium-expert-v2
  walker2d-medium-expert-v2
)

for seed in 2 4 6 8 10; do
  for env in "${ENVS[@]}"; do
    python active_iql_jax.py \
      --config_path "$CONFIG" \
      --gym_id "$env" \
      --seed "$seed"
  done
done
