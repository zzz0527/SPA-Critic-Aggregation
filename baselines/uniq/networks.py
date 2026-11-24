"""Flax networks used by the UNIQ reproduction."""

from typing import Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp


def orthogonal(scale: float = 2.0 ** 0.5):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for width in self.hidden_dims:
            x = nn.Dense(width, kernel_init=orthogonal())(x)
            x = nn.relu(x)
        return x


class ValueNetwork(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = MLP(self.hidden_dims)(observations)
        return jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(x), axis=-1)


class CriticNetwork(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate((observations, actions), axis=-1)
        x = MLP(self.hidden_dims)(x)
        return jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(x), axis=-1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            CriticNetwork(self.hidden_dims, name="q1")(observations, actions),
            CriticNetwork(self.hidden_dims, name="q2")(observations, actions),
        )


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray):
        x = MLP(self.hidden_dims)(observations)
        mean = nn.tanh(nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x))
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std


def gaussian_log_prob(
    mean: jnp.ndarray, log_std: jnp.ndarray, actions: jnp.ndarray
) -> jnp.ndarray:
    inv_std = jnp.exp(-log_std)
    quadratic = jnp.square((actions - mean) * inv_std)
    return -0.5 * jnp.sum(
        quadratic + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1
    )
