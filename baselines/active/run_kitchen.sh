CONFIG=configs/active_iql/kitchen_config.yaml
ENVS=(
  kitchen-complete-v0
  kitchen-partial-v0
  kitchen-mixed-v0
)

for seed in 2 4 6 8 10; do
  for env in "${ENVS[@]}"; do
    python active_iql_jax.py \
      --config_path "$CONFIG" \
      --gym_id "$env" \
      --seed "$seed"
  done
done
