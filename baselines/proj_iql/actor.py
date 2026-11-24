from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
           behavior_policy: Model,
           batch: Batch, temperature: float) -> Tuple[Model, InfoDict]:

    v = value(batch.observations)
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    adv = q - v

    exp_a = jnp.exp(adv * temperature)
    exp_a = jnp.clip(exp_a, 0.0, 100.0)

    beta_dist = behavior_policy(batch.observations)
    beta_log_probs = beta_dist.log_prob(batch.actions)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)

        # Support-constraint ratio is detached as in PIQL practical implementation.
        log_probs_detached = jax.lax.stop_gradient(log_probs)
        is_ratio = jnp.exp(log_probs_detached - beta_log_probs)
        is_ratio = jnp.clip(is_ratio, 0.0, 100.0)

        combined_weights = exp_a * is_ratio
        clipped_weights = jnp.minimum(combined_weights, 100.0)
        weights = jax.lax.stop_gradient(clipped_weights)

        actor_loss = -(weights * log_probs).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'is_ratio': is_ratio.mean(),
            'combined_weights': combined_weights.mean(),
            'weights': weights.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
