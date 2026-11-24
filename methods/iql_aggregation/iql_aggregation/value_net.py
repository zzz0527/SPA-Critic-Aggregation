from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from .common import MLP


class ValueCritic4Head(nn.Module):
    """Value network with the SPA heads used by the aggregation algorithm.

    The configurable trunk produces the state representation ``h_psi(s)`` and
    the ``expectile`` head predicts the usual IQL value ``V_psi(s)``. Three
    independent one-hidden-layer SPA regressors consume that representation:
    ``mar`` predicts total residual magnitude, while ``mar_up`` and
    ``mar_down`` predict its positive- and negative-residual partitions.
    """

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    mar_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    eps: float = 1e-6

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations
        for i, h in enumerate(self.hidden_dims):
            x = nn.Dense(h, name=f"trunk_{i}")(x)
            x = self.activations(x)

        exp_head = nn.Dense(1, name="expectile")(x)

        # SPA module heads share the value trunk h_psi(s) but keep separate
        # regressors for total, upper-side, and lower-side residual magnitudes.
        mar_hidden = nn.Dense(64, name="mar_hidden")(x)
        mar_hidden = self.activations(mar_hidden)
        mar_head = nn.Dense(1, name="mar")(mar_hidden)

        mar_up_hidden = nn.Dense(64, name="mar_up_hidden")(x)
        mar_up_hidden = self.activations(mar_up_hidden)
        marp_head = nn.Dense(1, name="mar_up")(mar_up_hidden)

        mar_down_hidden = nn.Dense(64, name="mar_down_hidden")(x)
        mar_down_hidden = self.activations(mar_down_hidden)
        marn_head = nn.Dense(1, name="mar_down")(mar_down_hidden)

        # Softplus enforces non-negative MAR estimates, matching the residual
        # magnitude interpretation in the split-point analysis.
        mar      = jnp.squeeze(self.mar_activation(mar_head), axis=-1)
        mar_up   = jnp.squeeze(self.mar_activation(marp_head), axis=-1)
        mar_down = jnp.squeeze(self.mar_activation(marn_head), axis=-1)
        expectile   = jnp.squeeze(exp_head, axis=-1)

        return {"mar": mar, "mar_up": mar_up, "mar_down": mar_down, "expectile": expectile}


class Critic(nn.Module):
    """Single state-action value network."""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    """Independent twin critics used to form a conservative Q estimate."""
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2
