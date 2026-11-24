"""Small, testable pieces of the UNIQ algorithm and task configurations."""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class UNIQConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    ensemble_size: int = 3
    nominal_expectiles: Tuple[float, ...] = (0.5, 0.7, 0.9)
    tau_min: float = 0.5
    tau_max: float = 0.9
    tau_beta: float = 5.0
    kappa: float = 0.5
    delta: float = 0.1
    epsilon: float = 1e-6
    discount: float = 0.99
    advantage_temperature: float = 3.0
    max_weight: float = 100.0
    learning_rate: float = 3e-4
    actor_learning_rate: Optional[float] = None
    target_value_decay: float = 0.995
    calibration_fraction: float = 0.2
    recalibration_interval: int = 5_000
    # The paper's main-text Algorithm 1 has a separate adaptive primary V,
    # whereas Appendix B.5.2 makes the nine ensemble members adaptive.  The
    # MuJoCo uses the latter; the additional AntMaze/Kitchen configurations use
    # the former.
    use_primary_value: bool = False

    @property
    def head_expectiles(self) -> Tuple[float, ...]:
        return tuple(
            tau
            for tau in self.nominal_expectiles
            for _ in range(self.ensemble_size)
        )

    @property
    def central_slice(self) -> slice:
        try:
            level = self.nominal_expectiles.index(0.7)
        except ValueError as exc:
            raise ValueError("UNIQ requires a nominal 0.7 expectile level") from exc
        start = level * self.ensemble_size
        return slice(start, start + self.ensemble_size)

    def validate(self) -> None:
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2 to estimate disagreement")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        if not 0.0 < self.tau_min <= self.tau_max <= 1.0:
            raise ValueError("expected 0 < tau_min <= tau_max <= 1")
        effective = np.asarray(self.head_expectiles)
        if not self.use_primary_value:
            effective = effective * self.tau_max
        if np.any(effective <= 0.0) or np.any(effective >= 1.0):
            raise ValueError("all effective expectiles must remain strictly inside (0, 1)")
        if self.actor_learning_rate is not None and self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive")


def config_for_env(
    env_name: str, enable_domain_primary: bool = False
) -> UNIQConfig:
    """Select the published MuJoCo or extended-domain configuration."""
    if enable_domain_primary and env_name.startswith("antmaze-"):
        # AntMaze IQL uses a high expectile and a much sharper AWR policy.
        # Keep the fixed ensemble for uncertainty, and adapt a separate primary
        # V directly in a high-expectile range so tau is never multiplied by
        # the nominal 0.7 level.
        return UNIQConfig(
            use_primary_value=True,
            tau_min=0.70,
            tau_max=0.95,
            kappa=0.0,
            advantage_temperature=10.0,
            actor_learning_rate=1e-3,
        )
    if enable_domain_primary and env_name.startswith("kitchen-"):
        # The Kitchen IQL baseline uses expectile 0.7 and policy
        # temperature 0.3.  The full [0.5, 0.9] adaptive range is centred at
        # 0.7 when normalized uncertainty is one.
        return UNIQConfig(
            use_primary_value=True,
            tau_min=0.50,
            tau_max=0.90,
            kappa=0.0,
            advantage_temperature=0.3,
            actor_learning_rate=3e-4,
        )

    # These are the only per-task assignments stated by the paper.
    config_a = {
        "halfcheetah-medium-expert-v2",
        "hopper-medium-replay-v2",
    }
    if env_name in config_a:
        return UNIQConfig(kappa=0.0, tau_max=0.95)
    return UNIQConfig(kappa=0.5, tau_max=0.90)


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    """Finite-sample split-conformal quantile from Eq. (27).

    The corrected rank is ceil((1-delta)(n+1)), clipped to n.  Scores are
    required to be finite and nonnegative because they are nonconformity
    magnitudes.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot calibrate from an empty score array")
    if not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must all be finite")
    if np.any(values < 0.0):
        raise ValueError("conformal scores must be nonnegative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")

    rank = int(np.ceil((1.0 - delta) * (values.size + 1)))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def adaptive_expectile_multiplier(
    disagreement: jnp.ndarray,
    q_hat: jnp.ndarray,
    tau_min: float,
    tau_max: float,
    beta: float,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute normalized uncertainty u(s) and Appendix Eq. (32)'s tau(s)."""
    uncertainty = disagreement / (q_hat + epsilon)
    multiplier = tau_min + (tau_max - tau_min) * jax_sigmoid(
        -beta * (uncertainty - 1.0)
    )
    return uncertainty, multiplier


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    # Written explicitly to keep this module's public math easy to unit test.
    return 1.0 / (1.0 + jnp.exp(-x))


def central_statistics(values: jnp.ndarray, central_slice: slice):
    """Mean and population std of the tau=0.7 ensemble heads.

    ``values`` has shape ``[num_heads, batch]``.
    """
    central = values[central_slice]
    return jnp.mean(central, axis=0), jnp.std(central, axis=0)
