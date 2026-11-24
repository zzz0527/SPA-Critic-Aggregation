CONFIG=configs/active_iql/antmaze_config.yaml
ENVS=(
  antmaze-umaze-v2
  antmaze-umaze-diverse-v2
  antmaze-medium-play-v2
  antmaze-medium-diverse-v2
  antmaze-large-play-v2
  antmaze-large-diverse-v2
)

for seed in 2 4 6 8 10; do
  for env in "${ENVS[@]}"; do
    python active_iql_jax.py \
      --config_path "$CONFIG" \
      --gym_id "$env" \
      --seed "$seed"
  done
done
