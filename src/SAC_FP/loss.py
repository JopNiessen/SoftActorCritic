"""
SAC loss functions
"""

# import libraries
import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
@eqx.filter_value_and_grad
def alpha_loss_fn(log_alpha, log_prob, target):
    return -jnp.mean(jnp.exp(log_alpha) * (log_prob + target))


@eqx.filter_jit
@eqx.filter_value_and_grad
def q_loss_fn(model, state, control, target):
    pred = jax.vmap(model)(state, control)
    return jnp.mean((pred - target)**2)


@eqx.filter_jit
@eqx.filter_value_and_grad
def v_loss_fn(model, state, target):
    pred = jax.vmap(model)(state)
    return jnp.mean((pred - target)**2)


@eqx.filter_jit
@eqx.filter_value_and_grad
def pi_loss_fn(model, state, alpha, v_pred, q1_fn, q2_fn, keys, control_scale=1):
    control, log_prob = jax.vmap(model)(state, keys)
    control = control * control_scale
    q1_pred = jax.vmap(q1_fn)(state, control)
    q2_pred = jax.vmap(q2_fn)(state, control)
    q_pred = jax.lax.min(q1_pred, q2_pred)
    q_pred = jnp.reshape(q_pred, v_pred.shape)
    advantage = q_pred - v_pred
    advantage = jnp.reshape(advantage, log_prob.shape)
    return jnp.mean(alpha * log_prob - advantage)
