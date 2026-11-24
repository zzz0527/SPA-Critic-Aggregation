"""IDQL with SPA routing over expectile-indexed critic groups.

The diffusion model proposes in-distribution actions. Each value ensemble
member summarizes one group of Q critics, and its SPA heads route those groups
when an action is selected at evaluation time.
"""
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import struct, traverse_util
from flax.serialization import from_bytes, to_bytes
import numpy as np
from idql_aggregation.data.dataset import DatasetDict
from idql_aggregation.networks import (
    DDPM,
    Ensemble,
    FourierFeatures,
    MLP,
    MLPResNet,
    StateActionValue,
    StateValue,
    cosine_beta_schedule,
    ddpm_sampler,
    vp_beta_schedule,
)
from idql_aggregation.spa import compute_spa_routing
from idql_aggregation.types import PRNGKey


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=self.rng)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)


def expectile_loss(diff, expectile=0.8):
    """Return the asymmetric squared error used by expectile regression."""
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def safe_mask_mean_batch(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute per-ensemble masked means, returning zero for empty subsets."""
    mask_f = mask.astype(values.dtype)
    denom = jnp.maximum(mask_f.sum(axis=-1), 1.0)
    return (values * mask_f).sum(axis=-1) / denom


def mask_grads_by_branch(grads: Dict, train_branch: str):
    """Zero gradients outside the requested expectile or SPA branch.

    Expectile updates train the shared trunk and V output; SPA updates train
    only the three MAR regressors.
    """
    flat = traverse_util.flatten_dict(grads)

    def keep(path_str: str) -> bool:
        if train_branch == "exp":
            return ("trunk" in path_str) or ("expectile" in path_str)
        if train_branch == "mar":
            return "mar" in path_str
        return True

    new_flat = {}
    for k, v in flat.items():
        path_str = "/".join(k)
        new_flat[k] = v if keep(path_str) else jnp.zeros_like(v)
    return traverse_util.unflatten_dict(new_flat)


def freeze_params_by_branch(new_params: Dict, old_params: Dict, train_branch: str):
    """Restore frozen parameters after applying an optimizer update.

    This complements gradient masking by blocking movement caused by optimizer
    state such as momentum.
    """
    old_flat = traverse_util.flatten_dict(old_params)
    new_flat = traverse_util.flatten_dict(new_params)

    def keep(path_str: str) -> bool:
        if train_branch == "exp":
            return ("trunk" in path_str) or ("expectile" in path_str)
        if train_branch == "mar":
            return "mar" in path_str
        return True

    merged_flat = {}
    for k, old_v in old_flat.items():
        path_str = "/".join(k)
        merged_flat[k] = new_flat[k] if keep(path_str) else old_v
    return traverse_util.unflatten_dict(merged_flat)


@partial(
    jax.jit,
    static_argnames=(
        "score_apply_fn",
        "critic_apply_fn",
        "value_apply_fn",
        "T",
        "act_dim",
        "N",
        "M",
        "clip_sampler",
        "num_value_qs",
        "qs_per_value",
        "eval_ensemble_method",
    ),
)
def _eval_actions_impl(
    rng,
    score_apply_fn,
    score_params,
    critic_apply_fn,
    critic_params,
    value_apply_fn,
    value_params,
    observations,
    T,
    act_dim,
    N,
    M,
    clip_sampler,
    alphas,
    alpha_hats,
    betas,
    ddpm_temperature,
    num_value_qs,
    qs_per_value,
    value_taus,
    soft_select_alpha,
    credibility_scale,
    eval_ensemble_method,
):
    """Select a diffusion candidate using SPA-routed critic aggregation.

    ``N`` candidates share one observation. The Q ensemble is partitioned into
    ``num_value_qs`` groups of ``qs_per_value`` critics, reduced within each
    group by a minimum, and then routed using the value ensemble's AS/SDS
    statistics. The same state-dependent routing weights score all candidates.
    """

    # Diffusion policy approximates the behavior distribution and proposes
    # in-support candidate actions a_1,...,a_N for the current state.
    obs_single = jnp.expand_dims(observations, axis=0)
    observations = obs_single.repeat(N, axis=0)

    actions, rng = ddpm_sampler(
        score_apply_fn,
        score_params,
        T,
        rng,
        act_dim,
        observations,
        alphas,
        alpha_hats,
        betas,
        ddpm_temperature,
        M,
        clip_sampler,
    )

    # Layout after reshaping: [value group, critics per group, candidate].
    q_ensemble = critic_apply_fn({"params": critic_params}, observations, actions)
    q_groups = q_ensemble.reshape(num_value_qs, qs_per_value, -1)
    q_group_min = q_groups.min(axis=1)

    if eval_ensemble_method in ("proxy", "soft_proxy"):
        # SPA module outputs z, z^+, z^- for each value expectile group.
        value_outputs = value_apply_fn({"params": value_params}, obs_single)
        mar = jnp.squeeze(value_outputs["mar"], axis=-1)
        mar_up = jnp.squeeze(value_outputs["mar_up"], axis=-1)
        mar_down = jnp.squeeze(value_outputs["mar_down"], axis=-1)
        routing = compute_spa_routing(
            mar,
            mar_up,
            mar_down,
            value_taus,
            temperature=soft_select_alpha,
            credibility_scale=credibility_scale,
        )
        routing_weight = routing["weights"]
        selected_ensemble_idx = jnp.argmax(routing_weight)

        if eval_ensemble_method == "soft_proxy":
            # Soft routing implements Q_agg(s,a)=sum_i w(s,i) Q_i(s,a).
            aggregated_q_values = (
                q_group_min * jnp.expand_dims(routing_weight, axis=-1)
            ).sum(axis=0)
        else:
            # Hard routing is the alpha->0 limit: use the best proxy-selected group.
            aggregated_q_values = q_group_min[selected_ensemble_idx]
    else:
        raise ValueError(f"Invalid eval_ensemble_method: {eval_ensemble_method}")

    # Eq. policy-select: choose the diffusion candidate with the largest
    # aggregated value under the routed critic bank.
    selected_action_idx = jnp.argmax(aggregated_q_values)
    return actions[selected_action_idx], rng, selected_ensemble_idx

class DDPMIQLLearner(Agent):
    """Train a diffusion behavior model and a SPA-routed IDQL critic bank.

    Value member ``i`` is paired with ``qs_per_value`` Q functions and expectile
    ``value_taus[i]``. Routing affects evaluation-time candidate selection only;
    actor fitting and Bellman targets retain the standard IDQL structure.
    """

    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    value_tx_exp: optax.GradientTransformation = struct.field(pytree_node=False)
    value_tx_mar: optax.GradientTransformation = struct.field(pytree_node=False)
    value_opt_state_exp: optax.OptState
    value_opt_state_mar: optax.OptState
    value_taus: jnp.ndarray
    soft_select_alpha: float
    credibility_scale: float
    last_ensemble_idx: jnp.ndarray
    num_value_qs: int = struct.field(pytree_node=False)
    qs_per_value: int = struct.field(pytree_node=False)
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    eval_ensemble_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    policy_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.7,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        num_value_qs: int = 4,
        value_taus: Optional[Sequence[float]] = [0.6,0.7,0.8,0.9],
        actor_num_blocks: int = 3,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = True,
        value_layer_norm: bool = False,
        policy_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 128,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(3e6),
        soft_select_alpha: float = 1.0,
        credibility_scale: float = 1.0,
        eval_ensemble_method: str = "soft_proxy",
    ):
        """Initialize diffusion, grouped-Q, value, and SPA model states."""

        if soft_select_alpha <= 0:
            raise ValueError("soft_select_alpha must be positive.")
        if credibility_scale <= 0:
            raise ValueError("credibility_scale must be positive.")

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish)

        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr))

        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        # Flatten the two logical critic axes for Ensemble; evaluation and
        # target construction reshape them back to [value group, Q replica].
        total_num_qs = num_value_qs * num_qs
        critic_def = Ensemble(critic_cls, num=total_num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_base_cls = partial(
            StateValue,
            hidden_dims=critic_hidden_dims,
            activations=nn.swish if value_layer_norm else nn.relu,
        )
        value_def = Ensemble(value_base_cls, num=num_value_qs)
        value_params = value_def.init(value_key, observations)["params"]
        value_tx_exp = optax.adam(learning_rate=value_lr)
        value_tx_mar = optax.adam(learning_rate=value_lr)

        # Value params are updated with split optimizer states: expectile
        # regression for V_{psi_i}, and SPA MAR regression for p_i(h; omega_i).
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        value_opt_state_exp = value_tx_exp.init(value_params)
        value_opt_state_mar = value_tx_mar.init(value_params)
        if value_taus is None:
            value_taus = jnp.full((num_value_qs,), critic_hyperparam)
        else:
            value_taus = jnp.asarray(value_taus)
            if value_taus.shape[0] != num_value_qs:
                raise ValueError("value_taus length must match num_value_qs.")
        if not bool(jnp.all((value_taus > 0.0) & (value_taus < 1.0))):
            raise ValueError("Every value expectile must lie strictly in (0, 1).")

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            value_tx_exp=value_tx_exp,
            value_tx_mar=value_tx_mar,
            value_opt_state_exp=value_opt_state_exp,
            value_opt_state_mar=value_opt_state_mar,
            value_taus=value_taus,
            soft_select_alpha=soft_select_alpha,
            credibility_scale=credibility_scale,
            last_ensemble_idx=jnp.array(-1, dtype=jnp.int32),
            num_value_qs=num_value_qs,
            qs_per_value=num_qs,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            critic_hyperparam=critic_hyperparam,
            clip_sampler=clip_sampler,
            policy_temperature=policy_temperature,
            eval_ensemble_method=eval_ensemble_method,
        )

    def _group_min_q(agent, batch: DatasetDict) -> jnp.ndarray:
        """Return the conservative Q target paired with each value member."""
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        return qs.reshape(agent.num_value_qs, agent.qs_per_value, -1).min(axis=1)

    def update_v_expectile(
        agent, batch: DatasetDict
    ) -> Tuple[Agent, Dict[str, float]]:
        """Update only the value trunk and expectile output."""
        q = agent._group_min_q(batch)

        def exp_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            outs = agent.value.apply_fn({"params": value_params}, batch["observations"])
            expectile = outs["expectile"]
            taus = jnp.expand_dims(agent.value_taus, axis=-1)
            loss_expectile_each = expectile_loss(q - expectile, taus).mean(axis=1)
            loss_expectile = loss_expectile_each.mean()
            return loss_expectile, {
                "loss_expectile": loss_expectile,
                "loss_expectile_std": jnp.std(loss_expectile_each),
                "v": expectile.mean(),
            }

        (loss, info), grads = jax.value_and_grad(
            exp_loss_fn, has_aux=True
        )(agent.value.params)
        grads = mask_grads_by_branch(grads, train_branch="exp")
        updates, opt_state = agent.value_tx_exp.update(
            grads,
            agent.value_opt_state_exp,
            agent.value.params,
        )
        params = optax.apply_updates(agent.value.params, updates)
        params = freeze_params_by_branch(
            params, agent.value.params, train_branch="exp"
        )
        value = agent.value.replace(params=params)
        return agent.replace(value=value, value_opt_state_exp=opt_state), {
            "value_loss": loss,
            **info,
        }

    def update_spa(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Fit only the post-hoc SPA heads with Q and V frozen."""
        q = agent._group_min_q(batch)

        def mar_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            outs = agent.value.apply_fn({"params": value_params}, batch["observations"])
            expectile = outs["expectile"]
            mar = outs["mar"]
            mar_up = outs["mar_up"]
            mar_down = outs["mar_down"]
            diff = jax.lax.stop_gradient(q - expectile)
            mar_target = jnp.abs(diff)
            mask_up = diff >= 0
            mask_down = diff < 0
            loss_mar_each = (
                jnp.mean((mar - mar_target) ** 2, axis=1)
                + safe_mask_mean_batch((mar_up - mar_target) ** 2, mask_up)
                + safe_mask_mean_batch((mar_down - mar_target) ** 2, mask_down)
            )
            loss_mar = loss_mar_each.mean()
            return loss_mar, {
                "loss_mar": loss_mar,
                "loss_mar_std": jnp.std(loss_mar_each),
                "mar": mar.mean(),
                "mar_up": mar_up.mean(),
                "mar_down": mar_down.mean(),
            }

        (loss, info), grads = jax.value_and_grad(
            mar_loss_fn, has_aux=True
        )(agent.value.params)
        grads = mask_grads_by_branch(grads, train_branch="mar")
        updates, opt_state = agent.value_tx_mar.update(
            grads,
            agent.value_opt_state_mar,
            agent.value.params,
        )
        updated_params = optax.apply_updates(agent.value.params, updates)
        params = freeze_params_by_branch(
            updated_params, agent.value.params, train_branch="mar"
        )
        value = agent.value.replace(params=params)
        return agent.replace(value=value, value_opt_state_mar=opt_state), {
            "spa_loss": loss,
            **info,
        }

    def update_v_joint(
        agent, batch: DatasetDict
    ) -> Tuple[Agent, Dict[str, float]]:
        """Optional joint variant: update V and then SPA on one batch."""
        agent, value_info = agent.update_v_expectile(batch)
        agent, spa_info = agent.update_spa(batch)
        return agent, {**value_info, **spa_info}

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Update every Q replica from its paired value member's target."""
        # Routing is intentionally absent from Bellman targets; it is used only
        # for evaluation-time policy extraction.
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )["expectile"]

        target_q_group = (
            jnp.expand_dims(batch["rewards"], axis=0)
            + agent.discount * jnp.expand_dims(batch["masks"], axis=0) * next_v
        )
        # Replicas in the same group share a value target but retain independent
        # parameters, enabling a conservative within-group minimum.
        target_q = jnp.repeat(target_q_group, repeats=agent.qs_per_value, axis=0)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss_each = ((qs - target_q) ** 2).mean(axis=1)
            critic_loss = critic_loss_each.mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "critic_loss_std": jnp.std(critic_loss_each),
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Fit the diffusion score model by noise prediction on dataset actions."""
        # SPA routing is applied later when selecting among sampled candidates.
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))

        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)

            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(
            score_model=score_model,
            target_score_model=target_score_model,
            rng=rng,
        )

        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        """Sample candidates and return the action preferred by SPA routing."""
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        action, rng, selected_ensemble_idx = _eval_actions_impl(
            self.rng,
            self.score_model.apply_fn,
            self.target_score_model.params,
            self.target_critic.apply_fn,
            self.target_critic.params,
            self.value.apply_fn,
            self.value.params,
            observations,
            self.T,
            self.act_dim,
            self.N,
            self.M,
            self.clip_sampler,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.ddpm_temperature,
            self.num_value_qs,
            self.qs_per_value,
            self.value_taus,
            self.soft_select_alpha,
            self.credibility_scale,
            self.eval_ensemble_method,
        )
        return np.array(action.squeeze()), self.replace(
            rng=rng, last_ensemble_idx=selected_ensemble_idx
        )

    @jax.jit
    def actor_update(self, batch: DatasetDict):
        """Run one actor-only update."""
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info

    @jax.jit
    def critic_update(self, batch: DatasetDict):
        """Train the expectile value bank and its paired Q critics."""
        new_agent = self
        new_agent, value_info = new_agent.update_v_expectile(batch)
        new_agent, critic_info = new_agent.update_q(batch)

        return new_agent, {**value_info, **critic_info}

    @jax.jit
    def critic_actor_update(self, batch: DatasetDict):
        """Synchronously train the critic/value bank and diffusion actor."""
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, value_info = new_agent.update_v_expectile(batch)
        new_agent, critic_info = new_agent.update_q(batch)

        return new_agent, {**actor_info, **value_info, **critic_info}

    @jax.jit
    def spa_update(self, batch: DatasetDict):
        """Fit the SPA module after the critic bank has been frozen."""
        return self.update_spa(batch)

    @jax.jit
    def update(self, batch: DatasetDict):
        """Synchronously update the actor, critic/value bank, and SPA heads."""
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, value_info = new_agent.update_v_joint(batch)
        new_agent, critic_info = new_agent.update_q(batch)

        return new_agent, {**actor_info, **critic_info, **value_info}

    def save(self, base_dir, seed: int):
        """Save every trainable and target state for this seed."""
        checkpoint_dir = Path(base_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "rng": self.rng,
            "score_model": self.score_model,
            "target_score_model": self.target_score_model,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "value": self.value,
            "value_opt_state_exp": self.value_opt_state_exp,
            "value_opt_state_mar": self.value_opt_state_mar,
        }
        with open(checkpoint_dir / f"seed_{seed}.ckpt", "wb") as checkpoint_file:
            checkpoint_file.write(to_bytes(payload))

    def load(self, base_dir, seed: int, load_spa: bool = False):
        """Restore actor/Q/V state, optionally restoring fitted SPA heads too."""
        checkpoint_path = Path(base_dir) / f"seed_{seed}.ckpt"
        template = {
            "rng": self.rng,
            "score_model": self.score_model,
            "target_score_model": self.target_score_model,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "value": self.value,
            "value_opt_state_exp": self.value_opt_state_exp,
            "value_opt_state_mar": self.value_opt_state_mar,
        }
        with open(checkpoint_path, "rb") as checkpoint_file:
            payload = from_bytes(template, checkpoint_file.read())

        value = payload["value"]
        value_opt_state_mar = payload["value_opt_state_mar"]
        if not load_spa:
            fresh_flat = traverse_util.flatten_dict(self.value.params)
            loaded_flat = traverse_util.flatten_dict(value.params)
            merged_flat = {
                key: (fresh_value if "mar" in "/".join(key) else loaded_flat[key])
                for key, fresh_value in fresh_flat.items()
            }
            value = value.replace(params=traverse_util.unflatten_dict(merged_flat))
            value_opt_state_mar = self.value_opt_state_mar

        return self.replace(
            rng=payload["rng"],
            score_model=payload["score_model"],
            target_score_model=payload["target_score_model"],
            critic=payload["critic"],
            target_critic=payload["target_critic"],
            value=value,
            value_opt_state_exp=payload["value_opt_state_exp"],
            value_opt_state_mar=value_opt_state_mar,
        )
