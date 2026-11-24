from typing import Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn

from .common import Batch, InfoDict, Model, Params, PRNGKey
from .spa import compute_spa_routing


def update_actor_aggregation(
        key: PRNGKey,
        actor: Model,
        critic_apply,
        critic_params,
        value_apply,
        value_params,
        batch: Batch,
        temperature: float,
        expectiles: jnp.ndarray,
        mode: int = 0,
        single_critic_index: int = 0,
        routing_alpha: float = 1.0,
        credibility_beta: float = 1.0,
    ) -> Tuple[Model, InfoDict, InfoDict]:
    """AWR actor update with SPA-based critic routing.

    MAR statistics and advantages use the layout ``[batch, critic]``. The
    default mode computes Credited-AS soft routing; modes 1--4 respectively
    use AS-only soft routing, uniform averaging, one fixed critic, and the
    critic with the largest per-state Credited-AS. Mode 5 independently shuffles
    the SPA-output/nominal-expectile tuples across critics for each state on
    every update. The resulting advantage is used only in the AWR objective.
    """

    # Step 2a: SPA module outputs z, z^+, z^- for every state and critic.
    vs = jax.vmap(lambda p: value_apply(p, batch.observations))(value_params)

    v_all = vs["expectile"].T
    mar_all = vs["mar"].T
    mar_up_all = vs["mar_up"].T
    mar_down_all = vs["mar_down"].T

    # Keep each diagnostic tuple paired with its nominal expectile. The Q/V
    # bank stays in its original order when the shuffled weights are applied.
    routing_expectiles = jnp.broadcast_to(expectiles, mar_all.shape)
    diagnostics = jnp.stack(
        (mar_all, mar_up_all, mar_down_all, routing_expectiles), axis=-1
    )

    def shuffle_diagnostics(operands):
        actor_key, outputs = operands
        actor_key, permutation_key = jax.random.split(actor_key)
        indices = jnp.broadcast_to(jnp.arange(outputs.shape[1]), outputs.shape[:2])
        permutation = jax.random.permutation(
            permutation_key, indices, axis=1, independent=True
        )
        return actor_key, jnp.take_along_axis(outputs, permutation[..., None], axis=1)

    key, diagnostics = jax.lax.cond(
        mode == 5, shuffle_diagnostics, lambda operands: operands, (key, diagnostics)
    )
    mar_all, mar_up_all, mar_down_all, routing_expectiles = (
        diagnostics[..., i] for i in range(4)
    )

    credited_routing = compute_spa_routing(
        mar_all,
        mar_up_all,
        mar_down_all,
        routing_expectiles,
        temperature=routing_alpha,
        credibility_scale=credibility_beta,
        use_credibility=True,
    )
    as_only_routing = compute_spa_routing(
        mar_all,
        mar_up_all,
        mar_down_all,
        routing_expectiles,
        temperature=routing_alpha,
        credibility_scale=credibility_beta,
        use_credibility=False,
    )

    # Mode 5 uses Credited-AS routing computed from the shuffled diagnostics.
    routing_weight = jnp.where(
        mode == 1,
        as_only_routing["weights"],
        credited_routing["weights"],
    )
    uniform_weight = jnp.full_like(
        routing_weight, 1.0 / routing_weight.shape[-1]
    )
    fixed_index = jnp.clip(
        jnp.asarray(single_critic_index), 0, routing_weight.shape[-1] - 1
    )
    fixed_weight = jnn.one_hot(
        fixed_index,
        routing_weight.shape[-1],
        dtype=routing_weight.dtype,
    )
    fixed_weight = jnp.broadcast_to(fixed_weight, routing_weight.shape)
    hard_weight = jnn.one_hot(
        jnp.argmax(credited_routing["credited_as"], axis=-1),
        routing_weight.shape[-1],
        dtype=routing_weight.dtype,
    )
    routing_weight = jnp.where(
        mode == 4,
        hard_weight,
        jnp.where(
            mode == 3,
            fixed_weight,
            jnp.where(mode == 2, uniform_weight, routing_weight),
        ),
    )

    as_score = credited_routing["as"]
    sds_score = credited_routing["sds"]
    routing_score = jnp.where(
        mode == 1,
        as_only_routing["standardized_signal"],
        credited_routing["standardized_signal"],
    )
    eligible = credited_routing["eligible"]

    sds_mean_per_head = sds_score.mean(axis=0)
    min_sds_index = jnp.argmin(sds_score, axis=1)
    as_mean_per_head = as_score.mean(axis=0)
    routing_score_mean_per_head = routing_score.mean(axis=0)
    max_as_index = jnp.argmax(as_score, axis=1)
    max_routing_score_index = jnp.argmax(routing_score, axis=1)
    selected_expectile = jnp.sum(routing_weight * expectiles, axis=-1)
    selected_index = jnp.argmax(routing_weight, axis=-1)

    # Step 2c/3: take the conservative twin-Q estimate for every head, then
    # aggregate Q_i-V_i at the advantage level for the AWR weight.
    q1, q2 = jax.vmap(lambda p: critic_apply(p, batch.observations, batch.actions))(critic_params)
    q1 = q1.T
    q2 = q2.T
    q_all = jnp.minimum(q1, q2)

    adv_all = q_all - v_all
    v_mean_per_head = v_all.mean(axis=0)
    q_mean_per_head = q_all.mean(axis=0)
    adv_mean_per_head = adv_all.mean(axis=0)
    adv_uniform = jnp.sum(uniform_weight * adv_all, axis=-1)
    adv_single = jnp.sum(fixed_weight * adv_all, axis=-1)
    adv_selected = jnp.sum(hard_weight * adv_all, axis=-1)
    adv_weighted = jnp.sum(routing_weight * adv_all, axis=-1)
    adv = adv_weighted
    adv *= temperature
    # Cap exponentiated advantages to prevent a few samples from dominating
    # the weighted behavior-cloning objective.
    exp_a = jnp.exp(adv)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Eq. policy-awr: maximize exp(t*A_agg) log pi(a|s); implemented as a
        # negative weighted log-likelihood minimization.
        dist = actor.apply(
            {'params': actor_params},
            batch.observations,
            training=True,
            rngs={'dropout': key},
        )
        log_probs = dist.log_prob(batch.actions)

        if log_probs.ndim > 1:
            log_probs = log_probs.sum(axis=-1)

        actor_loss = -(exp_a * log_probs).mean()

        info: InfoDict = {
            'actor_loss': actor_loss,
            'adv_weighted': adv_weighted.mean(),
            'adv_uniform': adv_uniform.mean(),
            'adv_single': adv_single.mean(),
            'adv_selected': adv_selected.mean(),
            'selected_index': selected_index.mean(),
            'selected_expectile': selected_expectile.mean(),
            'selected_expectile_std': selected_expectile.std(),
            'eligible_fraction': eligible.astype(jnp.float32).mean(),
            'min_SDS_index': min_sds_index.mean(),
            'max_AS_index': max_as_index.mean(),
            'max_routing_score_index': max_routing_score_index.mean(),
            **{f"v{i + 1}": v_mean_per_head[i] for i in range(v_mean_per_head.shape[0])},
            **{f"q{i + 1}": q_mean_per_head[i] for i in range(q_mean_per_head.shape[0])},
            **{f"adv{i + 1}": adv_mean_per_head[i] for i in range(adv_mean_per_head.shape[0])},
            **{f"SDS{i + 1}": sds_mean_per_head[i] for i in range(sds_mean_per_head.shape[0])},
            **{f"AS{i + 1}": as_mean_per_head[i] for i in range(as_mean_per_head.shape[0])},
            **{
                f"routing_score{i + 1}": routing_score_mean_per_head[i]
                for i in range(routing_score_mean_per_head.shape[0])
            },
        }
        return actor_loss, info

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    adv_info: InfoDict = {'adv_distribution': adv}

    return new_actor, info, adv_info
