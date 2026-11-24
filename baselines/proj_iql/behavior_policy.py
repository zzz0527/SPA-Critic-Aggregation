from typing import Tuple

import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params


def update(behavior_policy: Model, batch: Batch) -> Tuple[Model, InfoDict]:

    def loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = behavior_policy.apply({'params': params}, batch.observations)
        log_probs = dist.log_prob(batch.actions)

        if log_probs.ndim > 1:
            log_probs = log_probs.sum(-1)

        bc_loss = -log_probs.mean()

        return bc_loss, {'bc_loss': bc_loss}

    new_behavior_policy, info = behavior_policy.apply_gradient(loss_fn)
    return new_behavior_policy, info
