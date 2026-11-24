"""JAX/Flax implementation of UNIQ's appendix and main-text variants."""

from pathlib import Path
from typing import Dict, Iterable

import flax
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from core import (
    UNIQConfig,
    adaptive_expectile_multiplier,
    central_statistics,
    conformal_quantile,
)
from data import ArrayDataset, Batch
from networks import DoubleCritic, GaussianPolicy, ValueNetwork, gaussian_log_prob


def _stack_pytrees(pytrees: Iterable):
    pytrees = tuple(pytrees)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pytrees)


def _value_predictions(value_def, params, observations):
    return jax.vmap(
        lambda single_params: value_def.apply(
            {"params": single_params}, observations
        )
    )(params)


def _central_predictions(value_def, params, observations, central_slice):
    """Evaluate only the tau=0.7 members instead of all nine value heads."""
    central_params = jax.tree_util.tree_map(
        lambda leaf: leaf[central_slice], params
    )
    values = _value_predictions(value_def, central_params, observations)
    return jnp.mean(values, axis=0), jnp.std(values, axis=0)


class UNIQAgent:
    def __init__(
        self,
        seed: int,
        observation_example: np.ndarray,
        action_example: np.ndarray,
        config: UNIQConfig,
    ):
        config.validate()
        self.config = config
        self.q_hat = 1.0
        self.step = 0

        observations = jnp.asarray(observation_example, dtype=jnp.float32)
        actions = jnp.asarray(action_example, dtype=jnp.float32)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("initialization examples must have shape [batch, feature]")

        rng = jax.random.PRNGKey(seed)
        if config.use_primary_value:
            rng, actor_key, critic_key, value_key, primary_key = jax.random.split(
                rng, 5
            )
        else:
            # Preserve the original MuJoCo initialization exactly.
            rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
            primary_key = None
        value_keys = jax.random.split(value_key, len(config.head_expectiles))

        self.value_def = ValueNetwork(config.hidden_dims)
        self.critic_def = DoubleCritic(config.hidden_dims)
        self.actor_def = GaussianPolicy(config.hidden_dims, actions.shape[-1])

        value_params = _stack_pytrees(
            self.value_def.init(key, observations)["params"] for key in value_keys
        )
        critic_params = self.critic_def.init(
            critic_key, observations, actions
        )["params"]
        actor_params = self.actor_def.init(actor_key, observations)["params"]

        self.value_state = TrainState.create(
            apply_fn=self.value_def.apply,
            params=value_params,
            tx=optax.adam(config.learning_rate),
        )
        self.critic_state = TrainState.create(
            apply_fn=self.critic_def.apply,
            params=critic_params,
            tx=optax.adam(config.learning_rate),
        )
        actor_learning_rate = config.actor_learning_rate or config.learning_rate
        self.actor_state = TrainState.create(
            apply_fn=self.actor_def.apply,
            params=actor_params,
            tx=optax.adam(actor_learning_rate),
        )
        if config.use_primary_value:
            primary_params = self.value_def.init(primary_key, observations)["params"]
            self.primary_value_state = TrainState.create(
                apply_fn=self.value_def.apply,
                params=primary_params,
                tx=optax.adam(config.learning_rate),
            )
        else:
            self.primary_value_state = None
        self.target_value_params = value_params

        head_expectiles = jnp.asarray(config.head_expectiles, dtype=jnp.float32)
        central_slice = config.central_slice

        def update_fn(
            value_state,
            primary_value_state,
            target_value_params,
            critic_state,
            actor_state,
            batch,
            q_hat,
        ):
            observations = batch.observations
            actions = batch.actions

            q1, q2 = self.critic_def.apply(
                {"params": critic_state.params}, observations, actions
            )
            q_data = jax.lax.stop_gradient(jnp.minimum(q1, q2))

            def value_loss_fn(params):
                values = _value_predictions(self.value_def, params, observations)
                _, disagreement = central_statistics(values, central_slice)
                uncertainty, tau_multiplier = adaptive_expectile_multiplier(
                    jax.lax.stop_gradient(disagreement),
                    q_hat,
                    config.tau_min,
                    config.tau_max,
                    config.tau_beta,
                    config.epsilon,
                )
                if config.use_primary_value:
                    # Algorithm 1: this ensemble is fixed and exists only to
                    # estimate value/disagreement for calibration and Q targets.
                    effective_tau = head_expectiles[:, None]
                else:
                    # Appendix B.5.2: tau(s) multiplies every nominal head.
                    effective_tau = head_expectiles[:, None] * tau_multiplier[None, :]
                residual = q_data[None, :] - values
                weights = jnp.where(residual > 0.0, effective_tau, 1.0 - effective_tau)
                loss = jnp.mean(weights * jnp.square(residual))
                return loss, (
                    jnp.mean(values[central_slice]),
                    jnp.mean(disagreement),
                    jnp.mean(uncertainty),
                    jnp.mean(tau_multiplier),
                    jnp.min(tau_multiplier),
                    jnp.max(tau_multiplier),
                )

            (value_loss, value_aux), value_grads = jax.value_and_grad(
                value_loss_fn, has_aux=True
            )(value_state.params)
            value_state = value_state.apply_gradients(grads=value_grads)

            if config.use_primary_value:
                central_value, current_disagreement = _central_predictions(
                    self.value_def,
                    value_state.params,
                    observations,
                    central_slice,
                )
                current_uncertainty, adaptive_tau = adaptive_expectile_multiplier(
                    jax.lax.stop_gradient(current_disagreement),
                    q_hat,
                    config.tau_min,
                    config.tau_max,
                    config.tau_beta,
                    config.epsilon,
                )
                adaptive_tau = jax.lax.stop_gradient(adaptive_tau)

                def primary_value_loss_fn(params):
                    primary_value = self.value_def.apply(
                        {"params": params}, observations
                    )
                    residual = q_data - primary_value
                    weights = jnp.where(
                        residual > 0.0, adaptive_tau, 1.0 - adaptive_tau
                    )
                    loss = jnp.mean(weights * jnp.square(residual))
                    return loss, jnp.mean(primary_value)

                (primary_value_loss, primary_value_mean), primary_grads = (
                    jax.value_and_grad(primary_value_loss_fn, has_aux=True)(
                        primary_value_state.params
                    )
                )
                primary_value_state = primary_value_state.apply_gradients(
                    grads=primary_grads
                )
                diagnostic_aux = (
                    jnp.mean(central_value),
                    jnp.mean(current_disagreement),
                    jnp.mean(current_uncertainty),
                    jnp.mean(adaptive_tau),
                    jnp.min(adaptive_tau),
                    jnp.max(adaptive_tau),
                )
                effective_expectile_mean = jnp.mean(adaptive_tau)
            else:
                primary_value_loss = jnp.asarray(0.0, dtype=jnp.float32)
                primary_value_mean = value_aux[0]
                diagnostic_aux = value_aux
                effective_expectile_mean = jnp.mean(
                    head_expectiles
                ) * value_aux[3]

            target_value_params = jax.tree_util.tree_map(
                lambda target, online: (
                    config.target_value_decay * target
                    + (1.0 - config.target_value_decay) * online
                ),
                target_value_params,
                value_state.params,
            )

            next_mean, next_disagreement = _central_predictions(
                self.value_def,
                target_value_params,
                batch.next_observations,
                central_slice,
            )
            pessimistic_next_value = next_mean - config.kappa * next_disagreement
            target_q = jax.lax.stop_gradient(
                batch.rewards
                + config.discount * batch.masks * pessimistic_next_value
            )

            def critic_loss_fn(params):
                pred_q1, pred_q2 = self.critic_def.apply(
                    {"params": params}, observations, actions
                )
                loss = jnp.mean(
                    jnp.square(pred_q1 - target_q) + jnp.square(pred_q2 - target_q)
                )
                return loss, (jnp.mean(pred_q1), jnp.mean(pred_q2), jnp.mean(target_q))

            (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            if config.use_primary_value:
                policy_v = self.value_def.apply(
                    {"params": primary_value_state.params}, observations
                )
            else:
                policy_v, _ = _central_predictions(
                    self.value_def,
                    value_state.params,
                    observations,
                    central_slice,
                )
            policy_q1, policy_q2 = self.critic_def.apply(
                {"params": critic_state.params}, observations, actions
            )
            advantage = jax.lax.stop_gradient(
                jnp.minimum(policy_q1, policy_q2) - policy_v
            )
            awr_weight = jnp.minimum(
                jnp.exp(config.advantage_temperature * advantage),
                config.max_weight,
            )

            def actor_loss_fn(params):
                mean, log_std = self.actor_def.apply(
                    {"params": params}, observations
                )
                log_probability = gaussian_log_prob(mean, log_std, actions)
                loss = -jnp.mean(awr_weight * log_probability)
                return loss, (jnp.mean(advantage), jnp.mean(awr_weight))

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_state.params)
            actor_state = actor_state.apply_gradients(grads=actor_grads)

            info = {
                "value_loss": value_loss,
                "value_mean_0.7": diagnostic_aux[0],
                "ensemble_std": diagnostic_aux[1],
                "normalized_uncertainty": diagnostic_aux[2],
                "tau_multiplier_mean": diagnostic_aux[3],
                "tau_multiplier_min": diagnostic_aux[4],
                "tau_multiplier_max": diagnostic_aux[5],
                "primary_value_loss": primary_value_loss,
                "primary_value_mean": primary_value_mean,
                "effective_expectile_mean": effective_expectile_mean,
                "critic_loss": critic_loss,
                "q1_mean": critic_aux[0],
                "q2_mean": critic_aux[1],
                "target_q_mean": critic_aux[2],
                "actor_loss": actor_loss,
                "advantage_mean": actor_aux[0],
                "awr_weight_mean": actor_aux[1],
                "q_hat": q_hat,
            }
            return (
                value_state,
                primary_value_state,
                target_value_params,
                critic_state,
                actor_state,
                info,
            )

        self._update_fn = jax.jit(update_fn)
        self._central_value_fn = jax.jit(
            lambda params, obs: _central_predictions(
                self.value_def, params, obs, central_slice
            )[0]
        )
        self._diagnostics_fn = jax.jit(
            lambda params, obs, q_hat: self._diagnostics_impl(
                params, obs, q_hat
            )
        )
        # Evaluation calls sample_actions once per environment step. Compiling
        # this function avoids eager Flax/JAX dispatch overhead at every step.
        self._deterministic_action_fn = jax.jit(
            lambda params, obs: self.actor_def.apply(
                {"params": params}, obs
            )[0]
        )

    def _diagnostics_impl(self, params, observations, q_hat):
        central_mean, disagreement = _central_predictions(
            self.value_def,
            params,
            observations,
            self.config.central_slice,
        )
        uncertainty, multiplier = adaptive_expectile_multiplier(
            disagreement,
            q_hat,
            self.config.tau_min,
            self.config.tau_max,
            self.config.tau_beta,
            self.config.epsilon,
        )
        return central_mean, disagreement, uncertainty, multiplier

    def update(self, batch: Batch) -> Dict[str, np.ndarray]:
        jax_batch = Batch(*(jnp.asarray(x) for x in batch))
        (
            self.value_state,
            self.primary_value_state,
            self.target_value_params,
            self.critic_state,
            self.actor_state,
            info,
        ) = self._update_fn(
            self.value_state,
            self.primary_value_state,
            self.target_value_params,
            self.critic_state,
            self.actor_state,
            jax_batch,
            jnp.asarray(self.q_hat, dtype=jnp.float32),
        )
        self.step += 1
        return jax.device_get(info)

    def recalibrate(
        self,
        calibration: ArrayDataset,
        chunk_size: int = 16_384,
        use_terminal_masks: bool = False,
    ) -> float:
        """Recompute q-hat on held-out transitions.

        Eq. (25) omits terminal masking, so ``use_terminal_masks`` defaults to
        False.  The absolute Bellman residual is used because the appendix
        later calls alpha a magnitude and split-conformal scores must be
        nonnegative.
        """
        all_scores = []
        data = calibration.batch
        for start in range(0, calibration.size, chunk_size):
            stop = min(start + chunk_size, calibration.size)
            observations = jnp.asarray(data.observations[start:stop])
            next_observations = jnp.asarray(data.next_observations[start:stop])
            values = np.asarray(
                self._central_value_fn(self.value_state.params, observations)
            )
            next_values = np.asarray(
                self._central_value_fn(self.value_state.params, next_observations)
            )
            continuation = data.masks[start:stop] if use_terminal_masks else 1.0
            residual = (
                data.rewards[start:stop]
                + self.config.discount * continuation * next_values
                - values
            )
            all_scores.append(np.abs(residual).astype(np.float64))

        scores = np.concatenate(all_scores)
        self.q_hat = conformal_quantile(scores, self.config.delta)
        return self.q_hat

    def diagnostics(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        outputs = self._diagnostics_fn(
            self.value_state.params,
            jnp.asarray(observations, dtype=jnp.float32),
            jnp.asarray(self.q_hat, dtype=jnp.float32),
        )
        names = ("value", "ensemble_std", "uncertainty", "tau_multiplier")
        return dict(zip(names, (np.asarray(x) for x in outputs)))

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 0.0
    ) -> np.ndarray:
        del temperature  # Evaluation follows the repository's deterministic protocol.
        observations = np.asarray(observations, dtype=np.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
        mean = self._deterministic_action_fn(
            self.actor_state.params, jnp.asarray(observations)
        )
        actions = np.clip(np.asarray(mean), -1.0, 1.0)
        return actions[0] if squeeze else actions

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "value_state": self.value_state,
            "primary_value_state": self.primary_value_state,
            "target_value_params": self.target_value_params,
            "critic_state": self.critic_state,
            "actor_state": self.actor_state,
            "q_hat": np.asarray(self.q_hat, dtype=np.float32),
            "step": np.asarray(self.step, dtype=np.int64),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(flax.serialization.to_bytes(payload))
        temporary.replace(path)

    def load(self, path: Path) -> None:
        template = {
            "value_state": self.value_state,
            "primary_value_state": self.primary_value_state,
            "target_value_params": self.target_value_params,
            "critic_state": self.critic_state,
            "actor_state": self.actor_state,
            "q_hat": np.asarray(self.q_hat, dtype=np.float32),
            "step": np.asarray(self.step, dtype=np.int64),
        }
        payload = flax.serialization.from_bytes(template, Path(path).read_bytes())
        self.value_state = payload["value_state"]
        self.primary_value_state = payload["primary_value_state"]
        self.target_value_params = payload["target_value_params"]
        self.critic_state = payload["critic_state"]
        self.actor_state = payload["actor_state"]
        self.q_hat = float(payload["q_hat"])
        self.step = int(payload["step"])
