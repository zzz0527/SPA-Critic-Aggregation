import dataclasses
import os
import sys
from collections import deque
from typing import Dict, List

import gym
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from absl import app, flags
from flax.core import frozen_dict
from ml_collections import config_flags
from tqdm import tqdm

from idql_baseline.agents import DDPMIQLLearner
from idql_baseline.data import ReplayBuffer
from idql_baseline.data.d4rl_datasets import D4RLDataset
from idql_baseline.evaluation import evaluate
from idql_baseline.wrappers import wrap_gym
from idql_baseline.wrappers.wandb_video import WANDBVideo


FLAGS = flags.FLAGS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "idql_config.py")

flags.DEFINE_string("env_name", "halfcheetah-medium-expert-v2", "Environment name.")
flags.DEFINE_integer(
    "seed",
    2,
    "Random seed. The reported sweep uses 2, 4, 6, 8, and 10.",
)
flags.DEFINE_string("project", "idql-baseline", "Weights & Biases project.")
flags.DEFINE_string("experiment_name", "ddpm_idql_baseline", "Experiment name prefix.")
flags.DEFINE_string("wandb_dir", "wandb", "Local W&B directory.")
flags.DEFINE_boolean("use_wandb", True, "Log training and evaluation metrics to W&B.")
flags.DEFINE_string("save_dir", "./runs/", "Directory for local evaluation score files.")
flags.DEFINE_integer("max_steps", 1000000, "Number of updates in each training phase.")
flags.DEFINE_integer("batch_size", 256, "Mini-batch size.")
flags.DEFINE_integer(
    "eval_episodes",
    0,
    "Evaluation episodes; 0 selects 100 for AntMaze and 10 otherwise.",
)
flags.DEFINE_integer("log_interval", 1000, "Training log interval.")
flags.DEFINE_integer("eval_interval", 250000, "Evaluation interval.")
flags.DEFINE_boolean("save_video", False, "Save W&B evaluation videos.")
flags.DEFINE_boolean("normalize_returns", True, "Normalize D4RL MuJoCo returns.")
flags.DEFINE_string("inference_Ns", "16,64,256", "Comma-separated eval-time diffusion sample counts.")
flags.DEFINE_integer("N", 64, "Training-time diffusion sample count.")
flags.DEFINE_integer("M", 0, "Repeated final diffusion sampling steps.")
flags.DEFINE_boolean("clip_sampler", True, "Clip diffusion samples.")
flags.DEFINE_float("filter_threshold", None, "Optional dataset filter threshold.")
flags.DEFINE_integer("take_top", None, "Optional number of top trajectories to keep.")
flags.DEFINE_integer("online_max_steps", 0, "Optional online fine-tuning steps.")
flags.DEFINE_integer("online_start_training", 0, "Online step to begin critic updates.")
flags.DEFINE_integer("online_eval_interval", 5000, "Online evaluation interval.")
flags.DEFINE_integer("online_eval_episodes", 10, "Online evaluation episodes.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bars.")
config_flags.DEFINE_config_file(
    "config",
    DEFAULT_CONFIG_PATH,
    "Path to the IDQL baseline hyperparameter config.",
    lock_config=False,
)


@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for key in batch1.keys():
        merge[key] = jnp.concatenate([batch1[key], batch2[key]], axis=0)
    return frozen_dict.freeze(merge)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def append_evaluation_result(path, history, recent_returns, step, score):
    """Append one IQL-style evaluation row and rewrite the score file."""
    recent_returns.append(score)
    recent = np.asarray(recent_returns)
    history.append([step, score, recent.mean(), recent.std(ddof=0)])
    np.savetxt(
        path,
        history,
        fmt=["%.1f", "%.6f", "%.6f", "%.6f"],
    )


def safe_agent_replace(curr_agent, replace_dict):
    if not dataclasses.is_dataclass(curr_agent):
        return curr_agent
    valid_fields = {field.name for field in dataclasses.fields(curr_agent)}
    filtered = {key: value for key, value in replace_dict.items() if key in valid_fields}
    return curr_agent.replace(**filtered) if filtered else curr_agent


def make_details() -> Dict:
    rl_config = dict(FLAGS.config)
    rl_config["N"] = FLAGS.N
    rl_config["M"] = FLAGS.M
    rl_config["clip_sampler"] = FLAGS.clip_sampler

    if "antmaze" in FLAGS.env_name:
        rl_config["critic_hyperparam"] = 0.9
    else:
        rl_config["critic_hyperparam"] = 0.7

    inference_variants = [
        {"N": n, "clip_sampler": FLAGS.clip_sampler, "M": FLAGS.M}
        for n in parse_int_list(FLAGS.inference_Ns)
    ]

    return dict(
        project=FLAGS.project,
        experiment_name=FLAGS.experiment_name,
        group=f"{FLAGS.experiment_name}_{FLAGS.env_name}",
        run_name=(
            f"{FLAGS.experiment_name}_{FLAGS.env_name}_seed{FLAGS.seed}"
        ),
        env_name=FLAGS.env_name,
        seed=FLAGS.seed,
        max_steps=FLAGS.max_steps,
        batch_size=FLAGS.batch_size,
        eval_episodes=(
            FLAGS.eval_episodes
            if FLAGS.eval_episodes > 0
            else (100 if "antmaze" in FLAGS.env_name else 10)
        ),
        log_interval=FLAGS.log_interval,
        eval_interval=FLAGS.eval_interval,
        save_video=FLAGS.save_video,
        filter_threshold=FLAGS.filter_threshold,
        take_top=FLAGS.take_top,
        online_max_steps=FLAGS.online_max_steps,
        online_start_training=FLAGS.online_start_training,
        online_eval_interval=FLAGS.online_eval_interval,
        online_eval_episodes=FLAGS.online_eval_episodes,
        normalize_returns=FLAGS.normalize_returns,
        use_wandb=FLAGS.use_wandb,
        wandb_dir=FLAGS.wandb_dir,
        save_dir=FLAGS.save_dir,
        training_time_inference_params={"N": FLAGS.N, "clip_sampler": FLAGS.clip_sampler, "M": FLAGS.M},
        inference_variants=inference_variants,
        rl_config=rl_config,
    )


def call_main(details):
    if details["save_video"] and not details["use_wandb"]:
        raise ValueError("--save_video=True requires --use_wandb=True.")

    run = None
    if details["use_wandb"]:
        os.makedirs(details["wandb_dir"], exist_ok=True)
        run = wandb.init(
            project=details["project"],
            name=details["run_name"],
            group=details["group"],
            reinit=True,
            dir=details["wandb_dir"],
            settings=wandb.Settings(console="off"),
            save_code=False,
        )
        run.config.update(details)

    save_dir = os.path.join(details["save_dir"], details["env_name"])
    os.makedirs(save_dir, exist_ok=True)
    inference_ns = [params["N"] for params in details["inference_variants"]]
    eval_returns = {n: [] for n in inference_ns}
    last_n_returns = {n: deque(maxlen=3) for n in inference_ns}
    online_eval_returns = {n: [] for n in inference_ns}
    online_last_n_returns = {n: deque(maxlen=3) for n in inference_ns}

    env = gym.make(details["env_name"])
    dataset = D4RLDataset(env)
    dataset.seed(details["seed"])

    env = wrap_gym(env)
    env.seed(details["seed"])
    if details["save_video"]:
        env = WANDBVideo(env)

    if details["take_top"] is not None or details["filter_threshold"] is not None:
        dataset.filter(take_top=details["take_top"], threshold=details["filter_threshold"])

    agent = DDPMIQLLearner.create(
        details["seed"],
        env.observation_space,
        env.action_space,
        **details["rl_config"],
    )

    if "antmaze" in details["env_name"]:
        dataset.dataset_dict["rewards"] -= 1.0
    elif (
        "halfcheetah" in details["env_name"]
        or "walker2d" in details["env_name"]
        or "hopper" in details["env_name"]
    ) and details["normalize_returns"]:
        dataset.normalize_returns()

    def run_evaluation(step, num_episodes, namespace="eval"):
        nonlocal agent
        histories = eval_returns if namespace == "eval" else online_eval_returns
        recent_histories = (
            last_n_returns if namespace == "eval" else online_last_n_returns
        )
        for inference_params in details["inference_variants"]:
            agent = safe_agent_replace(agent, inference_params)
            eval_info = evaluate(
                agent,
                env,
                num_episodes,
                save_video=details["save_video"],
            )
            eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            n = inference_params["N"]
            filename = (
                f"{details['seed']}_N{n}.txt"
                if namespace == "eval"
                else f"{details['seed']}_online_N{n}.txt"
            )
            append_evaluation_result(
                os.path.join(save_dir, filename),
                histories[n],
                recent_histories[n],
                step,
                eval_info["return"],
            )
            if run is not None:
                run.log(
                    {
                        f"{namespace}/{inference_params}_{key}": value
                        for key, value in eval_info.items()
                    },
                    step=step,
                )
        agent = safe_agent_replace(agent, details["training_time_inference_params"])

    def offline_step(update_fn, step, phase, evaluate_policy=False):
        nonlocal agent
        sample = dataset.sample_jax(details["batch_size"])
        agent, info = update_fn(sample)

        if step % details["log_interval"] == 0 and run is not None:
            run.log(
                {f"train/{phase}/{key}": value for key, value in info.items()},
                step=step,
            )

        if evaluate_policy and step % details["eval_interval"] == 0:
            run_evaluation(step, details["eval_episodes"])

    disable_tqdm = not FLAGS.tqdm or not sys.stdout.isatty()
    for step in tqdm(range(1, details["max_steps"] + 1), smoothing=0.1, disable=disable_tqdm):
        offline_step(agent.update, step, "offline", evaluate_policy=True)

    if details["online_max_steps"] > 0:
        online_replay_buffer = ReplayBuffer(
            env.observation_space,
            env.action_space,
            details["online_max_steps"],
        )
        online_replay_buffer.seed(details["seed"] + 1241)

        observation, done = env.reset(), False
        for step in tqdm(
            range(1, details["online_max_steps"]),
            smoothing=0.1,
            disable=disable_tqdm,
        ):
            action, agent = agent.eval_actions(observation)
            next_observation, reward, done, info = env.step(action)

            mask = 1.0 if (not done or "TimeLimit.truncated" in info) else 0.0
            if "antmaze" in details["env_name"]:
                reward -= 1.0

            online_replay_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                )
            )
            observation = next_observation

            if done:
                observation, done = env.reset(), False

            if step > details["online_start_training"]:
                online_batch = online_replay_buffer.sample(128)
                offline_batch = dataset.sample(128)
                batch = merge_batch(online_batch, offline_batch)
                agent, info = agent.critic_update(batch)

                if step % details["log_interval"] == 0 and run is not None:
                    info = jax.device_get(info)
                    run.log(
                        {f"online_train/{key}": value for key, value in info.items()},
                        step=step + details["max_steps"],
                    )

                if step % details["online_eval_interval"] == 0:
                    run_evaluation(
                        step + details["max_steps"],
                        details["online_eval_episodes"],
                        namespace="online_eval",
                    )

    if run is not None:
        run.finish()


def main(_):
    call_main(make_details())


if __name__ == "__main__":
    app.run(main)
