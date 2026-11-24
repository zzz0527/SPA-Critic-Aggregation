"""Train the IQL-based UNIQ reproduction on D4RL datasets."""

import argparse
from collections import deque
import dataclasses
import json
from pathlib import Path
import random
import sys
from typing import Dict

import gym
import numpy as np
from tqdm import trange

import wrappers
from evaluation import evaluate

from agent import UNIQAgent
from core import config_for_env
from data import load_d4rl_dataset, split_train_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="halfcheetah-medium-v2")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=0, help="0 selects 100 for AntMaze and 10 otherwise.")
    parser.add_argument("--eval-interval", type=int, default=0, help="0 selects 100000 for AntMaze and 10000 otherwise.")
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--calibration-chunk-size", type=int, default=16_384)
    parser.add_argument(
        "--calibration-use-terminal-masks",
        action="store_true",
        help="Use terminal masks in conformal residuals (paper Eq. 25 omits them).",
    )
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--tau-max", type=float, default=None)
    parser.add_argument(
        "--domain-primary",
        action="store_true",
        help=(
            "Use the explicit main-text primary-value extrapolation for "
            "AntMaze/Kitchen; off by default because the paper did not test it."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def _write_json_line(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = config_for_env(
        args.env_name, enable_domain_primary=args.domain_primary
    )
    overrides = {}
    if args.kappa is not None:
        overrides["kappa"] = args.kappa
    if args.tau_max is not None:
        overrides["tau_max"] = args.tau_max
    if overrides:
        config = dataclasses.replace(config, **overrides)
    config.validate()

    # d4rl import registers its environments with Gym.
    import d4rl  # noqa: F401

    env = gym.make(args.env_name)
    # Keep the wrapper order identical to train_offline.make_env_and_dataset.
    # EpisodeMonitor converts D4RL returns to normalized scores before the
    # shared evaluation function reads info['episode']['return'].
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    try:
        env.seed(args.seed)
    except AttributeError:
        pass
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    full_dataset = load_d4rl_dataset(env, args.env_name)
    train_dataset, calibration_dataset = split_train_calibration(
        full_dataset, config.calibration_fraction, args.seed
    )
    sample_rng = np.random.RandomState(args.seed)

    agent = UNIQAgent(
        args.seed,
        train_dataset.batch.observations[:1],
        train_dataset.batch.actions[:1],
        config,
    )

    eval_episodes = args.eval_episodes if args.eval_episodes > 0 else (100 if "antmaze" in args.env_name else 10)
    eval_interval = args.eval_interval if args.eval_interval > 0 else (100_000 if "antmaze" in args.env_name else 10_000)

    run_dir = args.output_root / args.env_name / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "UNIQ",
        "paper": "UNIQ: Conformal Calibration for Adaptive Conservatism in Offline RL",
        "env_name": args.env_name,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_episodes": eval_episodes,
        "eval_interval": eval_interval,
        "train_size": train_dataset.size,
        "calibration_size": calibration_dataset.size,
        "config": dataclasses.asdict(config),
        "implementation": (
            "main_text_algorithm_1_primary_value_out_of_paper_domain_config"
            if config.use_primary_value
            else "appendix_eq_33_36"
        ),
        "domain_primary_requested": args.domain_primary,
        "absolute_conformal_residual": True,
        "calibration_uses_terminal_masks": args.calibration_use_terminal_masks,
        "evaluation_implementation": "repository evaluation.evaluate",
        "evaluation_return_semantics": "D4RL normalized score from EpisodeMonitor",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    writer = None
    if not args.no_tensorboard:
        from tensorboardX import SummaryWriter

        # Match train_offline.py: event files live directly in <root>/<env>/<seed>.
        writer = SummaryWriter(str(run_dir))

    q_hat = agent.recalibrate(
        calibration_dataset,
        chunk_size=args.calibration_chunk_size,
        use_terminal_masks=args.calibration_use_terminal_masks,
    )
    _write_json_line(run_dir / "metrics.jsonl", {"step": 0, "q_hat": q_hat})

    evaluation_rows = []
    last_three_returns = deque(maxlen=3)
    iterator = trange(
        1,
        args.max_steps + 1,
        smoothing=0.1,
        # Avoid writing carriage-return progress updates to non-interactive logs.
        disable=args.no_progress or not sys.stdout.isatty(),
        desc=f"UNIQ {args.env_name} seed={args.seed}",
    )
    for step in iterator:
        batch = train_dataset.sample(args.batch_size, sample_rng)
        update_info = agent.update(batch)

        if step % config.recalibration_interval == 0:
            q_hat = agent.recalibrate(
                calibration_dataset,
                chunk_size=args.calibration_chunk_size,
                use_terminal_masks=args.calibration_use_terminal_masks,
            )

        if step % args.log_interval == 0:
            train_record = {"step": step, "q_hat": float(agent.q_hat)}
            train_record.update({key: float(value) for key, value in update_info.items()})
            _write_json_line(run_dir / "metrics.jsonl", train_record)
            if writer is not None:
                for key, value in train_record.items():
                    if key != "step":
                        writer.add_scalar(f"training/{key}", value, step)

        if step % eval_interval == 0 or step == args.max_steps:
            eval_stats = evaluate(agent, env, eval_episodes)
            eval_record = {"step": step, **{f"eval_{key}": value for key, value in eval_stats.items()}}
            _write_json_line(run_dir / "metrics.jsonl", eval_record)
            last_three_returns.append(eval_stats["return"])
            recent = np.asarray(last_three_returns)
            evaluation_rows.append(
                (step, eval_stats["return"], float(np.mean(recent)), float(np.std(recent)))
            )
            np.savetxt(
                args.output_root / args.env_name / f"{args.seed}.txt",
                np.asarray(evaluation_rows),
                fmt=["%.1f", "%.6f", "%.6f", "%.6f"],
            )
            if writer is not None:
                writer.add_scalar("evaluation/average_returns", eval_stats["return"], step)
                writer.add_scalar("evaluation/average_lengths", eval_stats["length"], step)
                writer.flush()
            iterator.set_postfix(return_mean=f"{eval_stats['return']:.1f}", q_hat=f"{agent.q_hat:.3g}")

        if step % args.checkpoint_interval == 0 or step == args.max_steps:
            agent.save(run_dir / "checkpoint.msgpack")

    if writer is not None:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
