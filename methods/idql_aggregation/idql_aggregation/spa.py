"""Paper-faithful split-point analysis scores and routing."""

from typing import Dict

import jax.nn as jnn
import jax.numpy as jnp


def compute_spa_scores(
    mar: jnp.ndarray,
    mar_up: jnp.ndarray,
    mar_down: jnp.ndarray,
    expectiles: jnp.ndarray,
    credibility_scale: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, jnp.ndarray]:
    """Compute Equations (7), (10), and (12) along the final axis."""
    mar = jnp.asarray(mar)
    mar_up = jnp.asarray(mar_up)
    mar_down = jnp.asarray(mar_down)
    expectiles = jnp.asarray(expectiles, dtype=mar.dtype)

    as_score = jnp.where(mar > eps, mar_down / mar, 0.0)
    left = expectiles * mar * (mar_up - mar_down)
    right = mar_down * (mar_up - mar)
    sds = jnp.abs(left - right) / jnp.maximum(
        jnp.abs(left) + jnp.abs(right), eps
    )

    as_score = jnp.nan_to_num(as_score, nan=0.0, posinf=0.0, neginf=0.0)
    sds = jnp.nan_to_num(sds, nan=0.0, posinf=1.0, neginf=1.0)
    credited_as = as_score * jnp.exp(-sds / credibility_scale)
    return {
        "as": as_score,
        "sds": sds,
        "credited_as": credited_as,
    }


def eligible_softmax_weights(
    routing_signal: jnp.ndarray,
    as_score: jnp.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, jnp.ndarray]:
    """Compute Equation (13) with the paper's eligibility and fallback."""
    routing_signal = jnp.asarray(routing_signal)
    as_score = jnp.asarray(as_score)
    eligible = as_score > 1.0
    eligible_f = eligible.astype(routing_signal.dtype)
    eligible_count = eligible_f.sum(axis=-1, keepdims=True)
    safe_count = jnp.maximum(eligible_count, 1.0)

    mean = (routing_signal * eligible_f).sum(axis=-1, keepdims=True) / safe_count
    centered = routing_signal - mean
    variance = ((centered**2) * eligible_f).sum(
        axis=-1, keepdims=True
    ) / safe_count
    standardized = centered / (jnp.sqrt(variance) + eps)
    logits = jnp.where(eligible, standardized / temperature, -jnp.inf)
    eligible_weights = jnn.softmax(logits, axis=-1)

    fallback_index = jnp.argmax(as_score, axis=-1)
    fallback_weights = jnn.one_hot(
        fallback_index,
        as_score.shape[-1],
        dtype=routing_signal.dtype,
    )
    weights = jnp.where(
        eligible_count > 0,
        eligible_weights,
        fallback_weights,
    )
    return {
        "weights": weights,
        "eligible": eligible,
        "standardized_signal": jnp.where(eligible, standardized, 0.0),
        "fallback_index": fallback_index,
    }


def compute_spa_routing(
    mar: jnp.ndarray,
    mar_up: jnp.ndarray,
    mar_down: jnp.ndarray,
    expectiles: jnp.ndarray,
    temperature: float = 1.0,
    credibility_scale: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, jnp.ndarray]:
    """Compute scores and Credited-AS routing for the IDQL critic bank."""
    scores = compute_spa_scores(
        mar,
        mar_up,
        mar_down,
        expectiles,
        credibility_scale=credibility_scale,
        eps=eps,
    )
    routing = eligible_softmax_weights(
        scores["credited_as"],
        scores["as"],
        temperature=temperature,
        eps=eps,
    )
    return {**scores, **routing, "routing_signal": scores["credited_as"]}
