from typing import Tuple, Dict
import jax
import jax.numpy as jnp

from .common import Batch, InfoDict, Model, Params

from flax import traverse_util
import optax

def expectile_loss(diff, expectile=0.8):
    """Return the asymmetric squared error used by expectile regression."""
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def safe_mask_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Average a masked residual subset, returning zero when it is empty."""
    num = jnp.sum(jnp.where(mask, values, 0.0))
    den = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)

    return num / den


def mask_grads_by_branch(grads: Dict, train_branch: str):
    """Zero gradients outside the requested expectile or SPA branch.

    Expectile updates train the shared state trunk and V head. SPA updates train
    only parameters whose path contains ``mar``, keeping the fitted value
    representation fixed.
    """
    flat = traverse_util.flatten_dict(grads, sep="/")

    def keep(path_str: str) -> bool:
        if train_branch in ("expectile", "exp"):
            return ("trunk" in path_str) or ("expectile" in path_str)
        elif train_branch == "mar":
            return "mar" in path_str
        else:
            return True

    new_flat = {}
    for k, v in flat.items():
        new_flat[k] = v if keep(k) else jnp.zeros_like(v)
    return traverse_util.unflatten_dict(new_flat, sep="/")


def freeze_params_by_branch(new_params: Dict, old_params: Dict, train_branch: str):
    """Restore inactive parameters after an optimizer update.

    This is a parameter-level safeguard in addition to gradient masking, which
    prevents optimizer state (for example momentum) from moving a frozen branch.
    """
    old_flat = traverse_util.flatten_dict(old_params, sep="/")
    new_flat = traverse_util.flatten_dict(new_params, sep="/")

    def keep(path_str: str) -> bool:
        if train_branch in ("expectile", "exp"):
            return ("trunk" in path_str) or ("expectile" in path_str)
        elif train_branch == "mar":
            return "mar" in path_str
        else:
            return True

    merged_flat = {}
    for k, old_v in old_flat.items():
        merged_flat[k] = new_flat[k] if keep(k) else old_v
    return traverse_util.unflatten_dict(merged_flat, sep="/")


def update_v_joint_parallel(
        value_apply,
        value_params,
        opt_state_exp,
        opt_state_mar,
        tx_exp,
        tx_mar,
        critic_apply,
        critic_params,
        batch,
        taus,
    ):
        """Update all value/SPA heads with separate optimizer passes.

        The leading tree axis indexes paired expectile critics. For each pair,
        the expectile trunk is updated first and the SPA heads are then fitted
        to residuals produced by that updated value network.
        """
        # The pessimistic minimum matches the twin-Q estimate used by IQL.
        q1, q2 = jax.vmap(
            lambda p: critic_apply(p, batch.observations, batch.actions)
        )(critic_params)

        q = jnp.minimum(q1, q2)

        def exp_loss_fn(params, q_k, tau):
            # IQL expectile regression: V_{psi_i} is fitted to the in-sample
            # critic values at the corresponding tau_i.
            outs = value_apply(params, batch.observations)
            expectile = outs["expectile"]

            loss_expectile = expectile_loss(q_k - expectile, tau).mean()
            return loss_expectile, {
                "loss_expectile": loss_expectile,
            }


        def mar_loss_fn(params, q_k):
            # SPA regression targets from Eq. (SPA head training):
            # e_i(s,a)=Q_i(s,a)-V_i(s), with z/z+/z- trained on |e_i|.
            outs = value_apply(params, batch.observations)
            mar       = outs["mar"]
            mar_up    = outs["mar_up"]
            mar_down  = outs["mar_down"]
            expectile = outs["expectile"]

            # SPA targets describe the current critic/value residual; stopping
            # gradients prevents this auxiliary loss from changing Q or V.
            diff = jax.lax.stop_gradient(q_k - expectile)

            mar_target = jnp.abs(diff)
            mask_up    = diff >= 0
            mask_down  = diff < 0

            loss_mar = (
                jnp.mean((mar - mar_target) ** 2)
                + safe_mask_mean((mar_up   - mar_target) ** 2, mask_up)
                + safe_mask_mean((mar_down - mar_target) ** 2, mask_down)
            )

            return loss_mar, {
                "loss_mar": loss_mar,
            }


        def single_update(params, opt_state_exp, opt_state_mar, q_k, tau):
            # Pass 1 updates the representation and expectile output only.
            old_params = params
            (_, exp_info), exp_grads = jax.value_and_grad(
                exp_loss_fn, has_aux=True
            )(params, q_k, tau)
            exp_grads = mask_grads_by_branch(exp_grads, train_branch="exp")
            exp_updates, new_opt_state_exp = tx_exp.update(exp_grads, opt_state_exp, params)
            params = optax.apply_updates(params, exp_updates)
            params = freeze_params_by_branch(params, old_params, train_branch="exp")

            # Pass 2 treats the updated representation as fixed and fits only
            # the three residual-summary heads.
            (_, mar_info), mar_grads = jax.value_and_grad(mar_loss_fn, has_aux=True)(params, q_k)
            mar_grads = mask_grads_by_branch(mar_grads, train_branch="mar")
            mar_updates, new_opt_state_mar = tx_mar.update(mar_grads, opt_state_mar, params)
            new_params = optax.apply_updates(params, mar_updates)
            new_params = freeze_params_by_branch(new_params, params, train_branch="mar")

            info = {**mar_info, **exp_info}
            return new_params, new_opt_state_exp, new_opt_state_mar, info


        new_params, new_opt_state_exp, new_opt_state_mar, infos = jax.vmap(
            single_update, in_axes=(0, 0, 0, 0, 0)
        )(
            value_params,
            opt_state_exp,
            opt_state_mar,
            q,
            taus
        )

        loss_expectile = infos["loss_expectile"]
        loss_mar = infos["loss_mar"]
        info2: InfoDict = {
            **{
                f"loss_expectile_{i + 1}": loss_expectile[i]
                for i in range(loss_expectile.shape[0])
            },
            **{f"loss_mar_{i + 1}": loss_mar[i] for i in range(loss_mar.shape[0])},
        }

        return new_params, new_opt_state_exp, new_opt_state_mar, info2


def update_exp_parallel(
        value_apply,
        value_params,
        opt_state_exp,
        tx_exp,
        critic_apply,
        critic_params,
        batch,
        taus,
    ):
        """Stagewise update of the expectile value branch for every critic."""
        q1, q2 = jax.vmap(
            lambda p: critic_apply(p, batch.observations, batch.actions)
        )(critic_params)

        q = jnp.minimum(q1, q2)


        def exp_loss_fn(params, q_k, tau):
            outs = value_apply(params, batch.observations)
            expectile = outs["expectile"]

            loss_expectile = expectile_loss(q_k - expectile, tau).mean()
            return loss_expectile, {
                "loss_expectile": loss_expectile,
            }

        def single_update(params, opt_state_exp, q_k, tau):

            (_, exp_info), exp_grads = jax.value_and_grad(
                exp_loss_fn, has_aux=True
            )(params, q_k, tau)
            exp_grads = mask_grads_by_branch(exp_grads, train_branch="exp")
            exp_updates, new_opt_state_exp = tx_exp.update(exp_grads, opt_state_exp, params)
            new_params = optax.apply_updates(params, exp_updates)


            info = {**exp_info}
            return new_params, new_opt_state_exp, info


        new_params, new_opt_state_exp, infos = jax.vmap(single_update,in_axes=(0, 0, 0, 0)
        )(
            value_params,
            opt_state_exp,
            q,
            taus
        )

        loss_expectile = infos["loss_expectile"]
        info2: InfoDict = {
            f"loss_expectile_{i + 1}": loss_expectile[i]
            for i in range(loss_expectile.shape[0])
        }

        return new_params, new_opt_state_exp, info2


def update_mar_parallel(
        value_apply,
        value_params,
        opt_state_mar,
        tx_mar,
        critic_apply,
        critic_params,
        batch,
    ):
        """Fit SPA residual heads while freezing the learned value branch."""
        q1, q2 = jax.vmap(
            lambda p: critic_apply(p, batch.observations, batch.actions)
        )(critic_params)

        q = jnp.minimum(q1, q2)


        def mar_loss_fn(params, q_k):
            outs = value_apply(params, batch.observations)
            mar       = outs["mar"]
            mar_up    = outs["mar_up"]
            mar_down  = outs["mar_down"]
            expectile = outs["expectile"]

            diff = jax.lax.stop_gradient(q_k - expectile)

            mar_target = jnp.abs(diff)
            mask_up    = diff >= 0
            mask_down  = diff < 0

            loss_mar = jnp.mean((mar - mar_target) ** 2)
            loss_mar_up = safe_mask_mean((mar_up - mar_target) ** 2, mask_up)
            loss_mar_down = safe_mask_mean((mar_down - mar_target) ** 2, mask_down)



            return loss_mar + loss_mar_up + loss_mar_down, {
                "loss_mar": loss_mar,
                "loss_mar_up": loss_mar_up,
                "loss_mar_down": loss_mar_down,
            }

        def single_update(params, opt_state_mar, q_k):
            (_, mar_info), mar_grads = jax.value_and_grad(mar_loss_fn, has_aux=True)(params, q_k)
            mar_grads = mask_grads_by_branch(mar_grads, train_branch="mar")
            mar_updates, new_opt_state_mar = tx_mar.update(mar_grads, opt_state_mar, params)
            new_params = optax.apply_updates(params, mar_updates)
            new_params = freeze_params_by_branch(new_params, params, train_branch="mar")

            info = {**mar_info}
            return new_params, new_opt_state_mar, info


        new_params, new_opt_state_mar, infos = jax.vmap(single_update,in_axes=(0, 0, 0)
        )(
            value_params,
            opt_state_mar,
            q
        )

        loss_mar = infos["loss_mar"]
        loss_mar_up = infos["loss_mar_up"]
        loss_mar_down = infos["loss_mar_down"]
        info2: InfoDict = {
            **{f"loss_mar_{i + 1}": loss_mar[i] for i in range(loss_mar.shape[0])},
            **{f"loss_mar_up_{i + 1}": loss_mar_up[i] for i in range(loss_mar_up.shape[0])},
            **{f"loss_mar_down_{i + 1}": loss_mar_down[i] for i in range(loss_mar_down.shape[0])},
        }

        return new_params, new_opt_state_mar, info2


def update_q_parallel(
    critic_apply,
    critic_params,
    critic_opt_states,
    critic_tx,
    value_apply,
    value_params,
    batch: Batch,
    discount: float,
):
    """Apply paired Bellman updates to the vmapped twin-critic bank.

    Critic ``i`` bootstraps from its matching expectile value ``V_i``. SPA
    routing is deliberately absent from these targets and is used only during
    policy extraction.
    """
    k_value = jax.tree_util.tree_leaves(value_params)[0].shape[0]
    k_critic = jax.tree_util.tree_leaves(critic_params)[0].shape[0]
    if k_value != k_critic:
        raise ValueError(
            f"Mismatched ensemble size: value K={k_value}, critic K={k_critic}"
        )
    value_outs = jax.vmap(
        lambda p: value_apply(p, batch.next_observations)
    )(value_params)

    next_v_all = value_outs["expectile"]

    rewards = batch.rewards[None, :]
    masks   = batch.masks[None, :]

    target_q_all = rewards + discount * masks * next_v_all
    def critic_loss_fn(params, target_q_k):
        q1, q2 = critic_apply(params, batch.observations, batch.actions)
        loss = ((q1 - target_q_k) ** 2 + (q2 - target_q_k) ** 2).mean()
        return loss, {"critic_loss": loss}

    def single_update(params, opt_state, target_q_k):
        (loss, info), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(params, target_q_k)
        updates, new_opt = critic_tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, info

    new_params, new_opt_states, infos = jax.vmap(
        single_update,
        in_axes=(0, 0, 0)
    )(critic_params, critic_opt_states, target_q_all)

    critic_losses = infos["critic_loss"]
    info2: InfoDict = {
        f"critic_loss_{i + 1}": critic_losses[i]
        for i in range(critic_losses.shape[0])
    }
    return new_params, new_opt_states, info2
