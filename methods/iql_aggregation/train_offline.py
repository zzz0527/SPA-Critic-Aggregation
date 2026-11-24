import os
from typing import Tuple
from collections import deque

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import sys
import random

from iql_aggregation import wrappers
from iql_aggregation.dataset_utils import D4RLDataset, split_into_trajectories
from iql_aggregation.evaluation import evaluate
from iql_aggregation.learner import Learner

FLAGS = flags.FLAGS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'configs', 'mujoco_config.py')

flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('base_dir', './base_model/', 'Base directory for saved critic models.')
flags.DEFINE_boolean('load', False, 'Whether to load a pre-trained model from base_dir.')
flags.DEFINE_boolean(
    'load_actor',
    False,
    'Also restore the actor checkpoint; disabled for post-hoc extraction.',
)
flags.DEFINE_boolean(
    'load_spa',
    False,
    'Restore SPA-head parameters instead of fitting new post-hoc heads.',
)
flags.DEFINE_boolean('update_mar', True,
                     'Whether to update MAR when training from a loaded model.')
flags.DEFINE_boolean('save', False, 'Whether to save the model during training.')
flags.DEFINE_enum(
    'training_strategy',
    'stagewise',
    ['joint', 'stagewise'],
    'Training strategy. joint updates expectile, MAR, actor, and critic in one loop. '
    'stagewise runs critic/expectile, MAR, then actor phases sequentially.')
flags.DEFINE_integer('seed', 2, 'Random seed. The reported sweep uses 2, 4, 6, 8, and 10.')
flags.DEFINE_integer(
    'eval_episodes',
    0,
    'Evaluation episodes; 0 selects 100 for AntMaze and 10 otherwise.',
)
flags.DEFINE_boolean('save_eval_video', False,
                     'Whether to save evaluation rollouts as videos.')
flags.DEFINE_string('eval_video_dir', '',
                    'Directory for evaluation videos. Defaults to <save_dir>/eval_videos.')
flags.DEFINE_integer('eval_video_fps', 30, 'FPS for evaluation videos.')
flags.DEFINE_string('eval_camera_name', 'track',
                    'Camera name used for eval video rendering. Use an empty '
                    'string for the default camera.')
flags.DEFINE_integer('eval_render_width', 640, 'Render width for eval videos.')
flags.DEFINE_integer('eval_render_height', 480, 'Render height for eval videos.')
flags.DEFINE_boolean('eval_follow_agent_camera', True,
                     'Force a top-down camera that follows the agent (works for many MuJoCo envs).')
flags.DEFINE_float('eval_follow_camera_distance', 18.0,
                   'Camera distance when eval_follow_agent_camera=True.')
flags.DEFINE_float('eval_follow_camera_elevation', -90.0,
                   'Camera elevation when eval_follow_agent_camera=True.')
flags.DEFINE_float('eval_follow_camera_azimuth', 90.0,
                   'Camera azimuth when eval_follow_agent_camera=True.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('eval_video_interval', 500000,
                     'Save eval videos every N training steps (when save_eval_video=True).')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of updates per critic, actor, or joint phase.')
flags.DEFINE_integer(
    'spa_steps', 200000, 'Number of updates when fitting SPA separately.', lower_bound=1)
flags.DEFINE_integer(
    'ablation_mode',
    0,
    'Actor update mode: 0=Credited AS, 1=AS only, 2=uniform aggregation, '
    '3=single critic, 4=critic with the largest per-state Credited AS, '
    '5=Shuffled Aggregation.')
flags.DEFINE_integer('single_critic_index', 0,
                     'Zero-based critic index used for every state when '
                     '--ablation_mode=3.')
flags.DEFINE_string('expectiles', '0.6,0.7,0.8,0.9',
                    'Comma-separated expectile group, e.g. 0.5,0.6,0.7,0.8,0.9,0.95.')
flags.DEFINE_float('routing_alpha', 1.0,
                   'Routing softmax temperature alpha for expectile critic weighting.')
flags.DEFINE_float('credibility_beta', 1.0,
                   'Credibility scale beta for the SDS reliability gate exp(-SDS / beta).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    DEFAULT_CONFIG_PATH,
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):
    """Apply the standard D4RL MuJoCo return-range reward scaling."""

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    """Create the wrapped evaluation environment and its offline dataset."""
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name)
    base_dir = os.path.join(FLAGS.base_dir, FLAGS.env_name)
    summary_writer = SummaryWriter(os.path.join(save_dir, str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    eval_episodes = (
        FLAGS.eval_episodes
        if FLAGS.eval_episodes > 0
        else (100 if 'antmaze' in FLAGS.env_name else 10)
    )

    kwargs = dict(FLAGS.config)
    if FLAGS.expectiles:
        kwargs['expectiles'] = tuple(
            float(x.strip()) for x in FLAGS.expectiles.split(',') if x.strip()
        )
    kwargs['actor_mode'] = FLAGS.ablation_mode
    kwargs['single_critic_index'] = FLAGS.single_critic_index
    kwargs['routing_alpha'] = FLAGS.routing_alpha
    kwargs['credibility_beta'] = FLAGS.credibility_beta
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    last_n_returns = deque(maxlen=3)

    if FLAGS.load:
        agent.load(
            base_dir,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            load_actor_checkpoint=FLAGS.load_actor,
            load_spa=FLAGS.load_spa,
        )

    def log_training(update_info, step, phase):
        if step % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                v_np = np.asarray(v)
                if v_np.ndim == 0:
                    summary_writer.add_scalar(f'training/{phase}/{k}', v_np, step)
                else:
                    v_flat = v_np.ravel()
                    v_flat = v_flat[np.isfinite(v_flat)]
                    if v_flat.size == 0:
                        continue
                    summary_writer.add_histogram(f'training/{phase}/{k}', v_flat, step)
            summary_writer.flush()

    def run_evaluation(step):
        if FLAGS.save:
            agent.save(base_dir)

        eval_video_dir = FLAGS.eval_video_dir
        if FLAGS.save_eval_video and not eval_video_dir:
            eval_video_dir = os.path.join(save_dir, 'eval_videos')
        should_record_video = (
            FLAGS.save_eval_video and
            (step % max(1, FLAGS.eval_video_interval) == 0))

        eval_stats = evaluate(
            agent,
            env,
            eval_episodes,
            record_video=should_record_video,
            video_dir=eval_video_dir,
            video_prefix=f"step_{step:08d}",
            video_fps=FLAGS.eval_video_fps,
            render_width=FLAGS.eval_render_width,
            render_height=FLAGS.eval_render_height,
            camera_name=FLAGS.eval_camera_name.strip() or None,
            follow_agent_camera=FLAGS.eval_follow_agent_camera,
            follow_camera_distance=FLAGS.eval_follow_camera_distance,
            follow_camera_elevation=FLAGS.eval_follow_camera_elevation,
            follow_camera_azimuth=FLAGS.eval_follow_camera_azimuth)

        for k, v in eval_stats.items():
            summary_writer.add_scalar(f'evaluation/average_{k}s', v, step)
        summary_writer.flush()

        last_n_returns.append(eval_stats['return'])
        arr = np.array(last_n_returns)
        eval_returns.append([step, eval_stats['return'], arr.mean(), arr.std(ddof=0)])

        np.savetxt(os.path.join(save_dir, f'{FLAGS.seed}.csv'),
                   eval_returns,
                   fmt=['%.1f', '%.6f', '%.6f', '%.6f'],
                   delimiter=',',
                   header='step,normalized_return,rolling_mean_3,rolling_std_3',
                   comments='')

    def train_phase(phase_name, update_fn, num_steps, step_offset=0, evaluate_policy=False):
        """Run one fixed-length optimization phase with a global log offset."""
        iterator = tqdm.tqdm(
            range(1, num_steps + 1),
            desc=phase_name,
            smoothing=0.1,
            disable=(not FLAGS.tqdm or not sys.stdout.isatty()))
        for local_step in iterator:
            batch = dataset.sample(FLAGS.batch_size)
            global_step = step_offset + local_step
            update_info = update_fn(batch)
            log_training(update_info, global_step, phase_name)

            if evaluate_policy and local_step % FLAGS.eval_interval == 0:
                run_evaluation(global_step)

    if FLAGS.load:
        step_offset = 0
        if FLAGS.update_mar:
            train_phase('spa', agent.update_mar, FLAGS.spa_steps)
            step_offset = FLAGS.spa_steps
        elif not FLAGS.load_spa:
            raise ValueError(
                '--load requires either --update_mar=True or --load_spa=True.'
            )
        train_phase(
            'actor',
            agent.update_actor,
            FLAGS.max_steps,
            step_offset=step_offset,
            evaluate_policy=True,
        )
    elif FLAGS.training_strategy == 'joint':
        # Joint training performs value/SPA, actor, and critic updates per batch.
        train_phase('joint', agent.update_jointly, FLAGS.max_steps, evaluate_policy=True)
    else:
        # Stagewise training first fixes Q/V, then fits SPA, and finally trains
        # the actor against the routed advantages.
        train_phase('critic', agent.update_critics, FLAGS.max_steps)
        train_phase('spa', agent.update_mar, FLAGS.spa_steps, step_offset=FLAGS.max_steps)
        train_phase(
            'actor',
            agent.update_actor,
            FLAGS.max_steps,
            step_offset=FLAGS.max_steps + FLAGS.spa_steps,
            evaluate_policy=True,
        )


if __name__ == '__main__':
    app.run(main)
