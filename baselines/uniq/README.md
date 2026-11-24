# UNIQ Baseline

This directory contains the UNIQ reproduction used in the comparison experiments. It is implemented by extending the IQL baseline in `baselines/iql`; no official UNIQ code release was used.

## Source

The algorithm follows the paper [UNIQ: Conformal Calibration for Adaptive Conservatism in Offline Reinforcement Learning](https://arxiv.org/abs/2606.07592), while the D4RL processing, evaluation protocol, network widths, optimizer settings, and domain-level IQL conventions are based on the repository's IQL baseline.

## Training

Run commands from `baselines/uniq`.

MuJoCo:

```bash
python train_offline.py \
  --env-name=hopper-medium-replay-v2 \
  --output-root=./runs/ \
  --seed=2
```

AntMaze:

```bash
python train_offline.py \
  --env-name=antmaze-medium-play-v2 \
  --domain-primary \
  --eval-episodes=100 \
  --eval-interval=100000 \
  --output-root=./runs/ \
  --seed=2
```

Kitchen:

```bash
python train_offline.py \
  --env-name=kitchen-mixed-v0 \
  --domain-primary \
  --output-root=./runs/ \
  --seed=2
```

## Domain settings

The nine MuJoCo tasks use the configurations reported by UNIQ. Because the UNIQ paper does not report AntMaze or Kitchen results, `--domain-primary` enables the IQL-based extension used for the unified comparison: AntMaze uses adaptive expectiles in `[0.70, 0.95]` with AWR inverse temperature 10, and Kitchen uses `[0.50, 0.90]` with AWR inverse temperature 0.3. These extensions are recorded in each run's `metadata.json` and are not presented as settings reported by the UNIQ paper.

UNIQ uses three value networks at each nominal expectile in `{0.5, 0.7, 0.9}`, a disjoint 80/20 training/calibration split, conformal recalibration every 5,000 updates, and batch size 256. The reported seeds are `2, 4, 6, 8, 10`.

## Outputs

Evaluation scores are written to `<output-root>/<env_name>/<seed>.txt`. Each row contains `step`, normalized return, the latest-three mean, and the latest-three population standard deviation, matching IQL and Projective-IQL. TensorBoard logs, JSON records, metadata, and checkpoints are written under `<output-root>/<env_name>/<seed>/`.
