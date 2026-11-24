import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Sequence
from dataclasses import dataclass
import random
import yaml

import d4rl
import gym
import numpy as np
import pyrallis
from tqdm import tqdm
import flax
import flax.linen as nn
import flax.training.checkpoints as ckpt
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training.train_state import TrainState
import distrax
from tensorboardX import SummaryWriter


@dataclass
class TrainArgs:
    # Experiment
    exp_name: str = "active_iql_jax"
    gym_id: str = "halfcheetah-medium-expert-v2"
    seed: int = 1
    log_dir: str = "runs"
    save: bool = False
    save_dir: str = "active_iql_jax_agent"
    normalize_reward: bool = True
    # IQL
    total_iterations: int = int(1e6)
    gamma: float = 0.99
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    opt_decay_schedule: str = "cosine"
    batch_size: int = 256
    expectile: float = 0.7
    temperature: float = 3.0
    polyak: float = 0.005
    eval_freq: int = int(5e3)
    eval_episodes: int = 10
    log_freq: int = 1000
    # SQL
    layer_norm: bool = False
    dropout_rate: float = None
    value_dropout_rate: float = None
    # ACTIVE
    num_ensemble: int = 5
    alpha_lr: float = 1e-5
    target_likelihood: float = 1.75
    # Augmentations
    quantile: float = 0.0
    quantile_boot: float = 0.0

    def __post_init__(self):
        self.exp_name = f"{self.exp_name}__{self.gym_id}"
        self.save_dir = f"{self.exp_name}__seed{self.seed}__agent"


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class ValueNetwork(nn.Module):
    layer_norm: bool = False
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((256, 256, 1), layer_norm=self.layer_norm, dropout_rate=self.dropout_rate)(observations)
        # return jnp.squeeze(critic, -1)
        return critic


class CriticNetwork(nn.Module):
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((256, 256, 1),
                     layer_norm=self.layer_norm,
                     activations=self.activations)(inputs)
        # return jnp.squeeze(critic, -1)
        return critic


class DoubleCriticNetwork(nn.Module):
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = CriticNetwork(activations=self.activations,
                                layer_norm=self.layer_norm)(observations, actions)
        critic2 = CriticNetwork(activations=self.activations,
                                layer_norm=self.layer_norm)(observations, actions)
        return critic1, critic2


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


EXP_ADV_MAX = 100.0
LOG_STD_MAX = 2.0
LOG_STD_MIN = -10.0


class Actor(nn.Module):
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, temperature: float = 1.0, training: bool = False):
        x = MLP((256, 256),
                activate_final=True,
                dropout_rate=self.dropout_rate)(x, training=training)
        mean = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        mean = nn.tanh(mean)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim, ))
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std) * temperature)
        return dist


class TargetTrainState(TrainState):
    target_params: flax.core.FrozenDict


class KeyTrainState(TrainState):
    key: jax.Array


class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray


class Dataset:
    def __init__(self):
        self.size = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.masks = None
        self.dones_float = None
        self.next_observations = None

    def load(self, env, eps=1e-5):
        dataset = d4rl.qlearning_dataset(env)

        # Clip to eps
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        # Compute dones_float
        dones_float = np.zeros_like(dataset["rewards"], dtype=np.float32)
        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset["observations"][i + 1] -
                              dataset["next_observations"][i]
                              ) > 1e-6 or dataset["terminals"][i] == 1.0:
                dones_float[i] = 1.0
            else:
                dones_float[i] = 0.0
        dones_float[-1] = 1.0

        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.rewards = dataset["rewards"].reshape(-1, 1)
        self.masks = (1.0 - dataset["terminals"]).reshape(-1, 1)
        self.dones_float = dones_float.reshape(-1, 1)
        self.next_observations = dataset["next_observations"]
        self.size = len(self.observations)

    def sample(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        data = (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.masks[idx],
            self.next_observations[idx],
        )
        return Batch(*data)


def split_into_trajectories(dataset):
    trajs = [[]]
    for i in tqdm(range(len(dataset.observations)), desc="split trajs"):
        trajs[-1].append((dataset.observations[i], dataset.actions[i], dataset.rewards[i],
                          dataset.masks[i], dataset.dones_float[i], dataset.next_observations[i]))
        if dataset.dones_float[i] == 1.0 and i + 1 < len(dataset.observations):
            trajs.append([])
    return trajs


def merge_trajectories(trajs):
    observations, actions, rewards, masks, dones_float, next_observations = [], [], [], [], [], []
    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)
    return tuple(map(np.stack, (observations, actions, rewards, masks, dones_float, next_observations)))


def normalize(dataset):
    trajs = split_into_trajectories(dataset)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew
        return episode_return

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


if __name__ == "__main__":
    # Logging setup
    args = pyrallis.parse(config_class=TrainArgs)
    print(vars(args))
    run_name = f"{args.exp_name}__{args.seed}"
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key, value_key, dropout_key = jax.random.split(key, 5)

    # Eval env setup
    env = make_env(args.gym_id, args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    observation = env.observation_space.sample()[np.newaxis]
    action = env.action_space.sample()[np.newaxis]

    # Agent setup

    # Actor setup
    if args.opt_decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-args.actor_lr, args.total_iterations)
        actor_optimizer = optax.chain(optax.scale_by_adam(),
                                      optax.scale_by_schedule(schedule_fn))
    else:
        actor_optimizer = optax.adam(learning_rate=args.actor_lr)

    actor = Actor(action_dim=np.prod(env.action_space.shape), dropout_rate=args.dropout_rate)
    actor_state = KeyTrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, observation),
        key=dropout_key,
        tx=actor_optimizer
    )

    # Value setup
    vf = ValueNetwork(layer_norm=args.layer_norm, dropout_rate=args.value_dropout_rate)

    def vf_init_fn(key):
        return TrainState.create(
            apply_fn=vf.apply,
            params=vf.init(key, observation),
            tx=optax.adam(learning_rate=args.value_lr),
        )

    def vf_predict_fn(vf_state, observation):
        return vf.apply(vf_state.params, observation)

    value_keys = jax.random.split(value_key, args.num_ensemble)
    parallel_vf_init_fn = jax.vmap(vf_init_fn)
    parallel_vf_predict_fn = jax.vmap(vf_predict_fn, in_axes=(0, None))
    vf_states = parallel_vf_init_fn(value_keys)

    # Critic setup
    qf = DoubleCriticNetwork()
    qf_state = TargetTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(critic_key, observation, action),
        target_params=qf.init(critic_key, observation, action),
        tx=optax.adam(learning_rate=args.critic_lr)
    )

    # Alpha setup (riql, different from sql alpha)
    # target_entropy = -float(action_dim)
    key, alpha_key = jax.random.split(key)
    log_alpha = Scalar(0.0)
    alpha_state = TrainState.create(
        apply_fn=None,
        params=log_alpha.init(alpha_key),
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    # Dataset setup
    dataset = Dataset()
    dataset.load(env)
    # Preprocessing
    if args.normalize_reward:
        if "antmaze" in args.gym_id:
            dataset.rewards -= 1.0
            # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
            # See https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py
        elif ("halfcheetah" in args.gym_id or "walker2d" in args.gym_id
              or "hopper" in args.gym_id):
            normalize(dataset)

    def asymmetric_l2_loss(diff, expectile=0.8):
        weight = jnp.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def update_vf(vf_state, qf_state, batch):
        q1, q2 = qf.apply(qf_state.target_params, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)

        def vf_loss_fn(params):
            v = vf.apply(params, batch.observations)
            vf_loss = asymmetric_l2_loss(q - v, args.expectile).mean()
            return vf_loss, {
                "vf_loss": vf_loss,
                "v": v.mean(),
            }

        (vf_loss, info), grads = jax.value_and_grad(vf_loss_fn, has_aux=True)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)
        return vf_state, info

    parallel_update_vf = jax.vmap(update_vf, in_axes=(0, None, None))

    def update_actor(key, actor_state, vf_states, qf_state, alpha_state, batch):
        dropout_train_key = jax.random.fold_in(key=actor_state.key, data=actor_state.step)
        key, subkey = jax.random.split(key)
        vs = parallel_vf_predict_fn(vf_states, batch.observations)
        # v = vs.min(axis=0)
        v = jnp.quantile(vs, args.quantile, axis=0)
        q1, q2 = qf.apply(qf_state.target_params, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        weight = jnp.exp((q - v) * args.temperature)
        weight = jnp.minimum(weight, EXP_ADV_MAX)
        weight /= weight.mean()

        def actor_loss_fn(params, alpha_params):
            # Sampled/MLE log_prob
            dist = actor.apply(params, batch.observations,
                               training=True, rngs={"dropout": dropout_train_key})
            actions = dist.sample(seed=subkey)
            mle_log_prob = dist.log_prob(batch.actions).reshape(-1, 1)
            # Alpha loss
            log_alp = log_alpha.apply(alpha_params)
            alpha_loss = (log_alp * jax.lax.stop_gradient(mle_log_prob - args.target_likelihood)).mean()
            alp = jnp.exp(log_alp)
            alp = jax.lax.stop_gradient(alp)
            # Actor loss
            qa1, qa2 = qf.apply(qf_state.target_params, batch.observations, actions)
            qa = jnp.minimum(qa1, qa2)
            actor_loss = -(qa + alp * (weight * mle_log_prob)).mean()
            # Total loss
            total_loss = actor_loss + alpha_loss
            return total_loss, {
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "riql_alpha": alp,
                "total_loss": total_loss,
                "policy_q": qa.mean(),
                "mle_log_prob": mle_log_prob.mean(),
            }

        (actor_loss, info), grads = jax.value_and_grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(actor_state.params, alpha_state.params)
        actor_grads, alpha_grads = grads
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        return key, actor_state, alpha_state, info

    def update_qf(vf_states, qf_state, batch):
        next_vs = parallel_vf_predict_fn(vf_states, batch.next_observations)
        # next_v = next_vs.min(0)
        next_v = jnp.quantile(next_vs, args.quantile_boot, axis=0)
        target_q = batch.rewards + args.gamma * batch.masks * next_v

        def qf_loss_fn(params):
            q1, q2 = qf.apply(params, batch.observations, batch.actions)
            qf_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
            return qf_loss, {
                "qf_loss": qf_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

        (qf_loss, info), grads = jax.value_and_grad(qf_loss_fn, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)
        return qf_state, info

    def update_target(qf_state):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * args.polyak + tp * (1 - args.polyak), qf_state.params,
            qf_state.target_params)
        return qf_state.replace(target_params=new_target_params)

    @jax.jit
    def update(key, actor_state, vf_states, qf_state, alpha_state, batch):
        vf_states, vf_info = parallel_update_vf(vf_states, qf_state, batch)
        key, actor_state, alpha_state, actor_info = update_actor(
            key, actor_state, vf_states, qf_state, alpha_state, batch)
        qf_state, qf_info = update_qf(vf_states, qf_state, batch)
        qf_state = update_target(qf_state)
        return key, actor_state, vf_states, qf_state, alpha_state, {
            **vf_info, **actor_info, **qf_info
        }

    @jax.jit
    def get_action(rng, actor_state, observation, temperature=1.0):
        dist = actor.apply(actor_state.params, observation, temperature)
        rng, key = jax.random.split(rng)
        action = dist.sample(seed=key)
        return rng, jnp.clip(action, -1, 1)

    # Main loop
    pbar = tqdm(range(args.total_iterations + 1), unit="iter", dynamic_ncols=True)
    for global_step in pbar:

        # Batch update
        batch = dataset.sample(batch_size=args.batch_size)
        key, actor_state, vf_states, qf_state, alpha_state, update_info = update(
            key, actor_state, vf_states, qf_state, alpha_state, batch
        )

        # Evaluation
        if global_step % args.eval_freq == 0:
            env.seed(args.seed)
            stats = {"return": [], "length": []}
            for _ in range(args.eval_episodes):
                obs, done = env.reset(), False
                while not done:
                    key, action = get_action(key, actor_state, obs, temperature=0.0)
                    action = np.asarray(action)
                    obs, reward, done, info = env.step(action)
                for k in stats.keys():
                    stats[k].append(info["episode"][k[0]])
            for k, v in stats.items():
                writer.add_scalar(f"charts/episodic_{k}", np.mean(v), global_step)
                if k == "return":
                    normalized_score = env.get_normalized_score(np.mean(v)) * 100
                    writer.add_scalar("charts/normalized_score", normalized_score, global_step)
                    pbar.set_description("score: {:.2f}".format(normalized_score))
            writer.flush()

        # Logging
        if global_step % args.log_freq == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    writer.add_scalar(f"losses/{k}", v, global_step)
                else:
                    writer.add_histogram(f"losses/{k}", v, global_step)
            writer.flush()

    env.close()
    writer.close()

    # Save agent
    def save_state(path, state, name):
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        save_state = jax.device_get(state)
        prefix = name + "_"
        ckpt.save_checkpoint(path, save_state, args.total_iterations, prefix=prefix, keep=5, overwrite=True,
                             orbax_checkpointer=ckptr)

    if args.save:
        path = os.path.join(os.getcwd(), args.save_dir)
        save_state(path, actor_state, "actor")
        save_state(path, vf_states, "vfs")
        save_state(path, qf_state, "qf")
        with open(os.path.join(path, "args.yaml"), "w+", encoding="utf-8") as f:
            yaml.dump(vars(args), f, allow_unicode=True)
