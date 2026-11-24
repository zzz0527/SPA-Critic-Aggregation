"""IQL-Agg learner orchestration and checkpoint utilities."""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.serialization import (
    from_bytes,
    from_state_dict,
    msgpack_restore,
    to_bytes,
    to_state_dict,
)

from . import policy, value_net
from .actor import update_actor_aggregation as awr_update_actor
from .common import Batch, InfoDict, Model, PRNGKey
from .critic import (
    update_exp_parallel,
    update_mar_parallel,
    update_q_parallel,
    update_v_joint_parallel,
)


def stack_trees(tree_list):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *tree_list)


def save_actor(actor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "params": actor.params,
        "opt_state": actor.opt_state,
    }

    with open(path, "wb") as f:
        f.write(to_bytes(payload))


def load_actor(actor, path):
    with open(path, "rb") as f:
        payload = from_bytes(
            {
                "params": actor.params,
                "opt_state": actor.opt_state,
            },
            f.read(),
        )

    actor = actor.replace(
        params=payload["params"],
        opt_state=payload["opt_state"],
    )

    return actor


def save_single_critic(params, opt_state, path):
    payload = {
        "params": params,
        "opt_state": opt_state,
    }

    with open(path, "wb") as f:
        f.write(to_bytes(payload))


def _checkpoint_name(prefix, expectile, seed=None):
    seed_suffix = "" if seed is None else f"_seed{seed}"
    return f"{prefix}_{expectile:.3f}{seed_suffix}.ckpt"


def _checkpoint_path(base_dir, prefix, expectile, seed=None):
    path = base_dir / _checkpoint_name(prefix, expectile, seed)
    if seed is not None and not path.exists():
        legacy_path = base_dir / _checkpoint_name(prefix, expectile)
        if legacy_path.exists():
            return legacy_path
    return path


def save_critic_pool(
    critic_params,
    critic_opt_states,
    base_dir,
    expectiles,
    seed=None,
):
    base_dir.mkdir(parents=True, exist_ok=True)

    num_critics = jax.tree_util.tree_leaves(critic_params)[0].shape[0]

    for k in range(num_critics):
        single_params = jax.tree_util.tree_map(lambda x: x[k], critic_params)
        single_opt = jax.tree_util.tree_map(
            lambda x: x[k], critic_opt_states
        )
        tau = expectiles[k]

        save_single_critic(
            single_params,
            single_opt,
            base_dir / _checkpoint_name("critic", tau, seed),
        )


def save_critic_at_index(
    critic_params,
    critic_opt_states,
    base_dir,
    index,
    expectile,
    seed=None,
):
    base_dir.mkdir(parents=True, exist_ok=True)

    single_params = jax.tree_util.tree_map(lambda x: x[index], critic_params)
    single_opt = jax.tree_util.tree_map(lambda x: x[index], critic_opt_states)

    save_single_critic(
        single_params,
        single_opt,
        base_dir / _checkpoint_name("critic", expectile, seed),
    )


def load_single_critic(init_params, init_opt_state, path):
    with open(path, "rb") as f:
        payload = from_bytes(
            {
                "params": init_params,
                "opt_state": init_opt_state,
            },
            f.read(),
        )

    return payload["params"], payload["opt_state"]


def build_critic_ensemble(
    indices,
    base_dir,
    critic_def,
    key,
    observations,
    actions,
    expectiles,
    seed=None,
):
    params_list = []
    opt_list = []

    tx = optax.adam(3e-4)

    for idx in indices:
        init_params = critic_def.init(key, observations, actions)
        init_opt = tx.init(init_params)
        path = _checkpoint_path(base_dir, "critic", expectiles[idx], seed)

        p, o = load_single_critic(init_params, init_opt, path)

        params_list.append(p)
        opt_list.append(o)

    stacked_params = stack_trees(params_list)
    stacked_opt = stack_trees(opt_list)

    return stacked_params, stacked_opt


def save_single_value(params, exp_opt_state, mar_opt_state, path):
    payload = {
        "params": params,
        "opt_state_exp": exp_opt_state,
        "opt_state_mar": mar_opt_state,
    }

    with open(path, "wb") as f:
        f.write(to_bytes(payload))


def _merge_value_params_trunk_expectile_only(
    template_params_state, loaded_params_state
):
    template_flat = traverse_util.flatten_dict(template_params_state)
    loaded_flat = (
        traverse_util.flatten_dict(loaded_params_state)
        if isinstance(loaded_params_state, dict)
        else {}
    )

    merged_flat = {}
    for k, v in template_flat.items():
        key_segments = [str(x) for x in k]
        allow_load = any(
            seg.startswith("trunk_") or seg == "expectile"
            for seg in key_segments
        )
        merged_flat[k] = loaded_flat[k] if allow_load and k in loaded_flat else v

    return traverse_util.unflatten_dict(merged_flat)


def save_value_pool(
    value_params,
    exp_opt_states,
    mar_opt_states,
    base_dir,
    expectiles,
    seed=None,
):
    base_dir.mkdir(parents=True, exist_ok=True)

    num_values = jax.tree_util.tree_leaves(value_params)[0].shape[0]

    for k in range(num_values):
        single_params = jax.tree_util.tree_map(lambda x: x[k], value_params)
        single_exp_opt_state = jax.tree_util.tree_map(lambda x: x[k], exp_opt_states)
        single_mar_opt_state = jax.tree_util.tree_map(lambda x: x[k], mar_opt_states)

        tau = expectiles[k]

        save_single_value(
            single_params,
            single_exp_opt_state,
            single_mar_opt_state,
            base_dir / _checkpoint_name("value", tau, seed),
        )


def save_value_at_index(
    value_params,
    exp_opt_states,
    mar_opt_states,
    base_dir,
    index,
    expectile,
    seed=None,
):
    base_dir.mkdir(parents=True, exist_ok=True)

    single_params = jax.tree_util.tree_map(lambda x: x[index], value_params)
    single_exp_opt_state = jax.tree_util.tree_map(lambda x: x[index], exp_opt_states)
    single_mar_opt_state = jax.tree_util.tree_map(lambda x: x[index], mar_opt_states)

    save_single_value(
        single_params,
        single_exp_opt_state,
        single_mar_opt_state,
        base_dir / _checkpoint_name("value", expectile, seed),
    )


def load_single_value(
    init_params,
    init_exp_opt_state,
    init_mar_opt_state,
    path,
    load_mode: str = "full",
):
    if load_mode not in ("partial", "full"):
        raise ValueError(
            f"Unsupported load_mode: {load_mode}. Use 'partial' or 'full'."
        )

    template = {
        "params": init_params,
        "opt_state_exp": init_exp_opt_state,
        "opt_state_mar": init_mar_opt_state,
    }

    with open(path, "rb") as f:
        raw_bytes = f.read()

    if load_mode == "full":
        payload = from_bytes(template, raw_bytes)
        return payload["params"], payload["opt_state_exp"], payload["opt_state_mar"]

    loaded_state = msgpack_restore(raw_bytes)
    loaded_params_state = (
        loaded_state.get("params", {})
        if isinstance(loaded_state, dict)
        else {}
    )
    init_params_state = to_state_dict(init_params)

    merged_params_state = _merge_value_params_trunk_expectile_only(
        init_params_state, loaded_params_state
    )
    merged_params = from_state_dict(init_params, merged_params_state)
    return merged_params, init_exp_opt_state, init_mar_opt_state

def build_value_ensemble(
    indices,
    base_dir,
    value_def,
    key,
    observations,
    expectiles,
    seed=None,
    load_mode="full",
):
    params_list = []
    opt_exp_list = []
    opt_mar_list = []

    tx_exp = optax.adam(3e-4)
    tx_mar = optax.adam(3e-4)

    template_params = value_def.init(key, observations)
    template_opt_exp = tx_exp.init(template_params)
    template_opt_mar = tx_mar.init(template_params)

    for idx in indices:
        path = _checkpoint_path(base_dir, "value", expectiles[idx], seed)

        p, o_exp, o_mar = load_single_value(
            template_params,
            template_opt_exp,
            template_opt_mar,
            path,
            load_mode=load_mode,
        )

        params_list.append(p)
        opt_exp_list.append(o_exp)
        opt_mar_list.append(o_mar)

    stacked_params = stack_trees(params_list)
    stacked_opt_exp = stack_trees(opt_exp_list)
    stacked_opt_mar = stack_trees(opt_mar_list)

    return stacked_params, stacked_opt_exp, stacked_opt_mar

def make_update_joint_jit(critic_def, value_def, critic_tx, tx_exp, tx_mar):
    """Build the compiled joint value/SPA, actor, and critic update."""
    critic_apply = critic_def.apply
    value_apply = value_def.apply

    @jax.jit
    def update_jit(
        rng,
        actor,
        critic_params,
        critic_opt_states,
        target_critic_params,
        value_params,
        opt_state_exp,
        opt_state_mar,
        batch,
        discount,
        tau,
        expectiles,
        temperature,
        actor_mode,
        single_critic_index,
        routing_alpha,
        credibility_beta,
    ):
        # Joint protocol: train SPA/value heads, route the actor update, then
        # update the Bellman critics. This mirrors the optional joint variant
        # described in the paper while keeping aggregation out of Q targets.
        (
            new_value_params,
            new_opt_state_exp,
            new_opt_state_mar,
            value_info,
        ) = update_v_joint_parallel(
            value_apply,
            value_params,
            opt_state_exp,
            opt_state_mar,
            tx_exp,
            tx_mar,
            critic_apply,
            target_critic_params,
            batch,
            expectiles,
        )
        key, rng = jax.random.split(rng)
        new_actor, actor_info, adv_info = awr_update_actor(
            key,
            actor,
            critic_apply,
            target_critic_params,
            value_apply,
            new_value_params,
            batch,
            temperature,
            expectiles,
            actor_mode,
            single_critic_index,
            routing_alpha,
            credibility_beta,
        )
        new_critic_params, new_critic_opt_states, critic_info = update_q_parallel(
            critic_apply,
            critic_params,
            critic_opt_states,
            critic_tx,
            value_apply,
            new_value_params,
            batch,
            discount,
        )
        new_target_critic_params = jax.tree_util.tree_map(
            lambda t, s: (1.0 - tau) * t + tau * s,
            target_critic_params,
            new_critic_params
        )

        info = {**critic_info, **value_info, **actor_info, **adv_info}

        return (
            rng,
            new_actor,
            new_critic_params,
            new_critic_opt_states,
            new_value_params,
            new_opt_state_exp,
            new_opt_state_mar,
            new_target_critic_params,
            info
        )
    return update_jit


def make_update_critics_jit(critic_def, value_def, critic_tx, tx_exp):
    """Build stagewise Step 1a: paired expectile-value and Q updates."""
    critic_apply = critic_def.apply
    value_apply  = value_def.apply

    @jax.jit
    def update_jit(
        critic_params,
        critic_opt_states,
        target_critic_params,
        value_params,
        opt_state_exp,
        batch,
        discount,
        tau,
        expectiles,
    ):
        # Stagewise protocol, Step 1a: update Q_i and V_i for the critic bank.
        new_value_params, new_opt_state_exp, value_info = update_exp_parallel(
            value_apply,
            value_params,
            opt_state_exp,
            tx_exp,
            critic_apply,
            target_critic_params,
            batch,
            expectiles,
        )
        new_critic_params, new_critic_opt_states, critic_info = update_q_parallel(
            critic_apply,
            critic_params,
            critic_opt_states,
            critic_tx,
            value_apply,
            new_value_params,
            batch,
            discount,
        )
        new_target_critic_params = jax.tree_util.tree_map(
            lambda t, s: (1.0 - tau) * t + tau * s,
            target_critic_params,
            new_critic_params
        )

        info = {**critic_info, **value_info}

        return (
            new_critic_params,
            new_critic_opt_states,
            new_value_params,
            new_opt_state_exp,
            new_target_critic_params,
            info
        )
    return update_jit


def make_update_mar_jit(critic_def, value_def, tx_mar):
    """Build stagewise Step 1b: fit SPA heads with value trunks frozen."""
    value_apply  = value_def.apply
    critic_apply = critic_def.apply

    @jax.jit
    def update_jit(
        target_critic_params,
        value_params,
        opt_state_mar,
        batch,
        expectiles,
    ):
        # Stagewise protocol, Step 1b: fit SPA MAR heads on the learned critics.
        new_value_params, new_opt_state_mar, value_info = update_mar_parallel(
            value_apply,
            value_params,
            opt_state_mar,
            tx_mar,
            critic_apply,
            target_critic_params,
            batch
        )

        return (
            new_value_params,
            new_opt_state_mar,
            value_info
        )
    return update_jit


def make_update_actor_jit(critic_def, value_def):
    """Build the compiled SPA-routed AWR policy update."""
    critic_apply = critic_def.apply
    value_apply  = value_def.apply

    @jax.jit
    def update_jit(
        rng,
        actor,
        target_critic_params,
        value_params,
        batch,
        expectiles,
        temperature,
        actor_mode,
        single_critic_index,
        routing_alpha,
        credibility_beta,
    ):
        # Algorithm Step 2/3: compute routing weights and run AWR with the
        # aggregated critic-induced advantage.
        key, rng = jax.random.split(rng)
        new_actor, actor_info, adv_info = awr_update_actor(
            key,
            actor,
            critic_apply,
            target_critic_params,
            value_apply,
            value_params,
            batch,
            temperature,
            expectiles,
            actor_mode,
            single_critic_index,
            routing_alpha,
            credibility_beta,
        )

        info = {**actor_info, **adv_info}

        return (
            rng,
            new_actor,
            info
        )
    return update_jit


class Learner(object):
    """Own the expectile-indexed critic bank, SPA value bank, and AWR actor.

    Parameter trees for Q and V carry a leading ensemble axis aligned with
    ``expectiles``. The class exposes joint and stagewise update entry points
    while keeping the routing calculation confined to actor optimization.
    """

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 expectiles: Optional[Sequence[float]] = None,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 actor_mode: int = 0,
                 single_critic_index: int = 0,
                 routing_alpha: float = 1.0,
                 credibility_beta: float = 1.0):
        if expectiles is None:
            expectiles = (expectile,)
        # Candidate critic bank I={tau_i}. Each value/critic parameter tree is
        # vmapped over this leading ensemble dimension.
        self.base_expectiles = jnp.asarray(expectiles, dtype=jnp.float32)
        self.expectiles = self.base_expectiles
        num_ensemble = int(self.expectiles.shape[0])
        if num_ensemble == 0:
            raise ValueError("At least one expectile critic is required.")
        if not bool(jnp.all((self.expectiles > 0.0) & (self.expectiles < 1.0))):
            raise ValueError("Every expectile must lie strictly in (0, 1).")
        if routing_alpha <= 0:
            raise ValueError("routing_alpha must be positive.")
        if credibility_beta <= 0:
            raise ValueError("credibility_beta must be positive.")

        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.actor_mode = actor_mode
        self.routing_alpha = routing_alpha
        self.credibility_beta = credibility_beta
        self.seed = seed
        if actor_mode == 3 and not 0 <= single_critic_index < num_ensemble:
            raise ValueError(
                f"single_critic_index must be in [0, {num_ensemble - 1}] when actor_mode=3, "
                f"got {single_critic_index}."
            )
        self.single_critic_index = int(single_critic_index)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        value_keys = jax.random.split(value_key, num_ensemble)
        critic_keys = jax.random.split(critic_key, num_ensemble)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)
        def critic_init_fn(key):
            return critic_def.init(key, observations, actions)

        critic_params = jax.vmap(critic_init_fn)(critic_keys)
        critic_tx = optax.adam(learning_rate=critic_lr)
        critic_opt_states = jax.vmap(critic_tx.init)(critic_params)

        target_critic_params = jax.tree_util.tree_map(lambda x: x.copy(), critic_params)


        self.critic_def = critic_def
        self.critic_params = critic_params
        self.critic_opt_states = critic_opt_states
        self.critic_tx = critic_tx
        self.target_critic_params = target_critic_params


        value_def = value_net.ValueCritic4Head(hidden_dims)
        def value_init_fn(key):
            return value_def.init(key, observations)

        value_params = jax.vmap(value_init_fn)(value_keys)

        tx_exp = optax.adam(value_lr)
        tx_mar = optax.adam(value_lr)
        opt_state_exp = jax.vmap(tx_exp.init)(value_params)
        opt_state_mar = jax.vmap(tx_mar.init)(value_params)

        self.value_def = value_def
        self.value_params = value_params
        self.opt_state_exp = opt_state_exp
        self.opt_state_mar = opt_state_mar
        self.tx_exp = tx_exp
        self.tx_mar = tx_mar
        self.num_ensemble = num_ensemble

        self.actor = actor
        self.rng = rng

        self.update_joint_jit = make_update_joint_jit(
            self.critic_def,
            self.value_def,
            self.critic_tx,
            self.tx_exp,
            self.tx_mar,
        )
        self.update_critics_jit = make_update_critics_jit(
            self.critic_def,
            self.value_def,
            self.critic_tx,
            self.tx_exp,
        )
        self.update_mar_jit = make_update_mar_jit(self.critic_def, self.value_def, self.tx_mar)
        self.update_actor_jit = make_update_actor_jit(self.critic_def, self.value_def)


    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update_jointly(self, batch: Batch) -> InfoDict:
        """Run one complete joint IQL-Agg training iteration."""
        (
            new_rng,
            new_actor,
            new_critic_params,
            new_critic_opt_states,
            new_value_params,
            new_opt_state_exp,
            new_opt_state_mar,
            new_target_critic_params,
            info
        ) = self.update_joint_jit(
            self.rng,
            self.actor,
            self.critic_params,
            self.critic_opt_states,
            self.target_critic_params,
            self.value_params,
            self.opt_state_exp,
            self.opt_state_mar,
            batch,
            self.discount,
            self.tau,
            self.expectiles,
            self.temperature,
            self.actor_mode,
            self.single_critic_index,
            self.routing_alpha,
            self.credibility_beta,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic_params = new_critic_params
        self.critic_opt_states = new_critic_opt_states
        self.value_params = new_value_params
        self.opt_state_exp = new_opt_state_exp
        self.opt_state_mar = new_opt_state_mar
        self.target_critic_params = new_target_critic_params
        return info

    def update_critics(self, batch: Batch) -> InfoDict:
        """Update paired expectile values, critics, and target critics."""
        (
            new_critic_params,
            new_critic_opt_states,
            new_value_params,
            new_opt_state_exp,
            new_target_critic_params,
            info
        ) = self.update_critics_jit(
            self.critic_params,
            self.critic_opt_states,
            self.target_critic_params,
            self.value_params,
            self.opt_state_exp,
            batch,
            self.discount,
            self.tau,
            self.expectiles,
        )

        self.critic_params = new_critic_params
        self.critic_opt_states = new_critic_opt_states
        self.value_params = new_value_params
        self.opt_state_exp = new_opt_state_exp
        self.target_critic_params = new_target_critic_params
        return info

    def update_mar(self, batch: Batch) -> InfoDict:
        """Fit the SPA heads without changing the value representation."""
        (
            new_value_params,
            new_opt_state_mar,
            info
        ) = self.update_mar_jit(
            self.target_critic_params,
            self.value_params,
            self.opt_state_mar,
            batch,
            self.expectiles,
        )

        self.value_params = new_value_params
        self.opt_state_mar = new_opt_state_mar
        return info

    def update_actor(self, batch: Batch) -> InfoDict:
        """Update the actor from SPA-routed aggregate advantages."""
        (
            new_rng,
            new_actor,
            info
        ) = self.update_actor_jit(
            self.rng,
            self.actor,
            self.target_critic_params,
            self.value_params,
            batch,
            self.expectiles,
            self.temperature,
            self.actor_mode,
            self.single_critic_index,
            self.routing_alpha,
            self.credibility_beta,
        )

        self.rng = new_rng
        self.actor = new_actor
        return info

    def save(self, base_dir):
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        save_actor(self.actor, base_dir / "actor.ckpt")
        if self.actor_mode == 3:
            save_critic_at_index(
                self.target_critic_params,
                self.critic_opt_states,
                base_dir / "critics",
                self.single_critic_index,
                float(self.expectiles[self.single_critic_index]),
                self.seed,
            )
            save_value_at_index(
                self.value_params,
                self.opt_state_exp,
                self.opt_state_mar,
                base_dir / "values",
                self.single_critic_index,
                float(self.expectiles[self.single_critic_index]),
                self.seed,
            )
        else:
            save_critic_pool(
                self.target_critic_params,
                self.critic_opt_states,
                base_dir / "critics",
                self.expectiles,
                self.seed,
            )
            save_value_pool(
                self.value_params,
                self.opt_state_exp,
                self.opt_state_mar,
                base_dir / "values",
                self.expectiles,
                self.seed,
            )

    def load(
        self,
        base_dir,
        obs_dim,
        action_dim,
        load_actor_checkpoint=False,
        load_spa=False,
    ):
        """Load a critic bank for post-hoc SPA fitting or full continuation."""
        base_dir = Path(base_dir)

        if load_actor_checkpoint:
            self.actor = load_actor(self.actor, base_dir / "actor.ckpt")
        self.target_critic_params, self.critic_opt_states = build_critic_ensemble(
            indices=range(int(self.base_expectiles.shape[0])),
            base_dir=base_dir / "critics",
            critic_def=self.critic_def,
            key=self.rng,
            observations=jnp.zeros((1, obs_dim.shape[-1])),
            actions=jnp.zeros((1, action_dim.shape[-1])),
            expectiles=self.base_expectiles,
            seed=self.seed,
        )
        self.value_params, self.opt_state_exp, self.opt_state_mar = build_value_ensemble(
            indices=range(int(self.base_expectiles.shape[0])),
            base_dir=base_dir / "values",
            value_def=self.value_def,
            key=self.rng,
            observations=jnp.zeros((1, obs_dim.shape[-1])),
            expectiles=self.base_expectiles,
            seed=self.seed,
            load_mode="full" if load_spa else "partial",
        )

        self.critic_params = jax.tree_util.tree_map(lambda x: x.copy(), self.target_critic_params)
