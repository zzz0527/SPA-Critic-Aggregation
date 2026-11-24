from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(critic: Model, value: Model, actor: Model,
             behavior_policy: Model, batch: Batch) -> Tuple[Model, InfoDict]:

    pi_phi_dist = actor(batch.observations)
    pi_phi_log_probs = pi_phi_dist.log_prob(batch.actions)
    pi_phi_probs = jnp.exp(pi_phi_log_probs)

    pi_beta_dist = behavior_policy(batch.observations)
    pi_beta_log_probs = pi_beta_dist.log_prob(batch.actions)
    pi_beta_probs = jnp.exp(pi_beta_log_probs)

    pi_phi_probs = jax.lax.stop_gradient(pi_phi_probs)
    pi_beta_probs = jax.lax.stop_gradient(pi_beta_probs)

    dot_beta_phi = jnp.dot(pi_beta_probs, pi_phi_probs)
    phi_sq = jnp.dot(pi_phi_probs, pi_phi_probs)
    scalar_proj = dot_beta_phi / (phi_sq + 1e-8)

    tau_proj_per_sample = scalar_proj * pi_phi_probs
    tau_proj_per_sample = jnp.clip(tau_proj_per_sample, 0.5, 1.0)
    tau_proj = jnp.mean(tau_proj_per_sample)

    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        value_loss = loss(q - v, tau_proj).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'tau_proj': tau_proj,
            'scalar_proj': scalar_proj,
            'pi_phi_probs': pi_phi_probs.mean(),
            'pi_beta_probs': pi_beta_probs.mean()
        }

    new_value, info = value.apply_gradient(value_loss_fn)
    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
