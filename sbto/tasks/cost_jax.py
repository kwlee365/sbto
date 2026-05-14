"""
JAX (device-side) versions of the cost kernels defined in cost.py.

Goal: keep rollout outputs (x, u, obs) on GPU and compute the per-sample cost
without round-tripping to CPU. Used by OCPBase.cost_jax (built lazily on first
call).

Kernels accept jax arrays. Reference and weight tensors are uploaded once at
task construction time. Each kernel slices ref/weight to var.shape[1] so it
works for partial-horizon rollouts (incremental opt) -- jit will retrace once
per distinct T value, same caching pattern as SimMjxRollout._rollout_fn.

The order of JAX_COST_FUNS must match COST_FUNS in cost.py so the integer
f_idx stored in OCPBase._cost_terms lines up.
"""
import jax.numpy as jnp


def quadratic_cost_jax(var, ref, weight):
    """
    var:    (N, T, I)
    ref:    (T_full, I)
    weight: (T_full, I)
    returns: (N,)
    """
    T = var.shape[1]
    diff = var - ref[:T][None, :, :]
    return jnp.sum(weight[:T][None, :, :] * diff * diff, axis=(1, 2))


def quaternion_dist_logmap_jax(var, ref, weight):
    """
    var:    (N, T, Nquat*4) -- flattened quaternions
    ref:    (T_full, Nquat*4)
    weight: (T_full, 1) (numba kernel only uses [:, 0])
    returns: (N,)
    """
    T = var.shape[1]
    Nq4 = var.shape[2]
    Nquat = Nq4 // 4
    v = var.reshape(var.shape[0], T, Nquat, 4)
    r = ref[:T].reshape(T, Nquat, 4)[None, :, :, :]
    dot = jnp.sum(v * r, axis=-1)              # (N, T, Nquat)
    dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
    angle = 2.0 * jnp.arccos(dot)              # (N, T, Nquat)
    w = weight[:T, 0][None, :, None]           # (1, T, 1)
    return jnp.sum(w * angle * angle, axis=(1, 2))


def hamming_dist_jax(cnt_rollout, cnt_plan, weight):
    """
    cnt_rollout: (N, T, C) -- may be float; matches numba (s > 1 -> 1)
    cnt_plan:    (T_full, C) -- int 0/1
    weight:      (T_full, C)
    returns: (N,)
    """
    T = cnt_rollout.shape[1]
    s = jnp.clip(cnt_rollout, 0.0, 1.0).astype(jnp.int32)
    plan = cnt_plan[:T].astype(jnp.int32)[None, :, :]
    diff = jnp.bitwise_xor(s, plan)
    return jnp.sum(weight[:T][None, :, :] * diff.astype(weight.dtype), axis=(1, 2))


# Index in this list must match COST_FUNS in cost.py so f_idx lines up.
JAX_COST_FUNS = [
    quadratic_cost_jax,           # 0 -- quadratic_cost_nb
    quaternion_dist_logmap_jax,   # 1 -- quaternion_dist_logmap_nb
    hamming_dist_jax,             # 2 -- hamming_dist_nb
]
