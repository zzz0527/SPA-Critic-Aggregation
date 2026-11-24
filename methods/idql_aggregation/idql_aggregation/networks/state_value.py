import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence

from idql_aggregation.networks import default_init


class StateValue(nn.Module):
    """State-value ensemble member with SPA residual-summary heads.

    The configurable trunk produces ``h_i(s)`` and ``expectile`` predicts
    ``V_i(s)``. Three independent one-hidden-layer SPA regressors consume that
    representation and estimate total, positive-side, and negative-side
    residual magnitudes used to compute AS, SDS, and routing weights.
    """

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    mar_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    mar_hidden_dim: int = 64
    eps: float = 1e-6

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs):
        x = observations
        for i, h in enumerate(self.hidden_dims):
            x = nn.Dense(
                h, kernel_init=default_init(), name=f"trunk_{i}"
            )(x)
            x = self.activations(x)

        exp_head = nn.Dense(
            1, kernel_init=default_init(), name="expectile"
        )(x)

        # SPA heads share the learned state representation h_i(s), but use
        # separate regressors for total/upper/lower residual magnitudes.
        mar_hidden = nn.Dense(
            self.mar_hidden_dim, kernel_init=default_init(), name="mar_hidden"
        )(x)
        mar_hidden = self.activations(mar_hidden)
        mar_head = nn.Dense(1, kernel_init=default_init(), name="mar")(mar_hidden)

        mar_up_hidden = nn.Dense(
            self.mar_hidden_dim, kernel_init=default_init(), name="mar_up_hidden"
        )(x)
        mar_up_hidden = self.activations(mar_up_hidden)
        marp_head = nn.Dense(1, kernel_init=default_init(), name="mar_up")(mar_up_hidden)

        mar_down_hidden = nn.Dense(
            self.mar_hidden_dim, kernel_init=default_init(), name="mar_down_hidden"
        )(x)
        mar_down_hidden = self.activations(mar_down_hidden)
        marn_head = nn.Dense(
            1, kernel_init=default_init(), name="mar_down"
        )(mar_down_hidden)

        # MAR quantities are residual magnitudes, so the heads are constrained
        # to be non-negative.
        mar = jnp.squeeze(self.mar_activation(mar_head), axis=-1)
        mar_up = jnp.squeeze(self.mar_activation(marp_head), axis=-1)
        mar_down = jnp.squeeze(self.mar_activation(marn_head), axis=-1)
        expectile = jnp.squeeze(exp_head, axis=-1)

        return {
            "mar": mar,
            "mar_up": mar_up,
            "mar_down": mar_down,
            "expectile": expectile,
        }
