"""
SAC training function
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from src.SAC_FP.loss import *
from src.SAC_FP.update import *


"""Regular training function"""
def train_step(params, buffer, batch_size, train_iter, key, **kwargs):
    gamma = kwargs.get('gamma', .9)
    policy_update_freq = kwargs.get('policy_update_freq', 2)

    # take sample from memory
    traj_obs, traj_control, obs, control, reward, next_obs, done = buffer.sample_from_buffer(batch_size)

    # define function approximators
    (pi, qf1, qf2, vf, vf_target, af) = params
    log_alpha, alpha_opt, alpha_state, alpha_target = af
    v_fn, _, _ = vf
    vt_fn = vf_target
    q1_fn, _, _ = qf1
    q2_fn, _, _ = qf2
    pi_fn, _, _ = pi

    keys = jrandom.split(key, len(obs))
    new_control, log_prob = jax.vmap(pi_fn)(obs, keys)
    
    # update alpha (dual problem)
    alpha_loss, alpha_grads = alpha_loss_fn(log_alpha, log_prob, alpha_target)
    log_alpha, alpha_opt, alpha_state = update_model((log_alpha, alpha_opt, alpha_state), alpha_grads)
    alpha = jnp.exp(log_alpha)

    af = (log_alpha, alpha_opt, alpha_state, alpha_target)
    
    # Q-function gradients
    mask = 1 - done
    v_target = jax.vmap(vt_fn)(next_obs)
    q_target = reward + gamma * v_target * mask
    q1_loss, q1_grads = q_loss_fn(q1_fn, obs, control, q_target)
    q2_loss, q2_grads = q_loss_fn(q2_fn, obs, control, q_target)
    
    # value function gradients
    v_pred = jax.vmap(v_fn)(obs)
    q1_pred = jax.vmap(q1_fn)(obs, new_control)
    q2_pred = jax.vmap(q2_fn)(obs, new_control)
    q_pred = jax.lax.min(q1_pred, q2_pred)
    v_target = q_pred - alpha * log_prob
    v_loss, v_grads = v_loss_fn(v_fn, obs, v_target)
    
    if train_iter % policy_update_freq == 0:
        # update actor
        pi_loss, pi_grads = pi_loss_fn(pi_fn, obs, alpha, v_pred, q1_fn, q2_fn, keys)
        pi = update_model(pi, pi_grads)

        # update value target
        vf_target = update_vf_target(v_fn, vt_fn)
    else:
        pi_loss = 0
        
    # update Q-functions
    qf1 = update_model(qf1, q1_grads)
    qf2 = update_model(qf2, q2_grads)

    q_loss = q1_loss + q2_loss

    # update value function
    vf = update_model(vf, v_grads)

    return (pi, qf1, qf2, vf, vf_target, af), (pi_loss, q_loss, v_loss, alpha_loss)


"""Recurrent training function"""
def train_step_rec(params, buffer, batch_size, train_iter, key, **kwargs):
    gamma = kwargs.get('gamma', .9)
    policy_update_freq = kwargs.get('policy_update_freq', 2)

    # take sample from memory
    traj_obs, traj_control, obs, control, reward, next_obs, done = buffer.sample_from_buffer(batch_size)

    # define function approximators
    (pi, qf1, qf2, vf, vf_target, af) = params
    log_alpha, alpha_opt, alpha_state, alpha_target = af
    v_fn, _, _ = vf
    vt_fn = vf_target
    q1_fn, _, _ = qf1
    q2_fn, _, _ = qf2
    pi_fn, _, _ = pi

    keys = jrandom.split(key, len(obs))
    new_control, log_prob = jax.vmap(pi_fn)(traj_obs, traj_control, keys)
    
    # update alpha (dual problem)
    alpha_loss, alpha_grads = alpha_loss_fn(log_alpha, log_prob, alpha_target)
    log_alpha, alpha_opt, alpha_state = update_model((log_alpha, alpha_opt, alpha_state), alpha_grads)
    alpha = jnp.exp(log_alpha)

    af = (log_alpha, alpha_opt, alpha_state, alpha_target)
    
    # Q-function gradients
    mask = 1 - done
    v_target = jax.vmap(vt_fn)(next_obs)
    q_target = reward + gamma * v_target * mask
    q1_loss, q1_grads = q_loss_fn(q1_fn, obs, control, q_target)
    q2_loss, q2_grads = q_loss_fn(q2_fn, obs, control, q_target)
    
    # value function gradients
    v_pred = jax.vmap(v_fn)(obs)
    q1_pred = jax.vmap(q1_fn)(obs, new_control)
    q2_pred = jax.vmap(q2_fn)(obs, new_control)
    q_pred = jax.lax.min(q1_pred, q2_pred)
    v_target = q_pred - alpha * log_prob
    v_loss, v_grads = v_loss_fn(v_fn, obs, v_target)
    
    if train_iter % policy_update_freq == 0:
        # update actor
        pi_loss, pi_grads = pi_rec_loss_fn(pi_fn, traj_obs, traj_control, alpha, v_pred, q1_fn, q2_fn, keys)
        pi = update_model(pi, pi_grads)

        # update value target
        vf_target = update_vf_target(v_fn, vt_fn)
    else:
        pi_loss = 0
        
    # update Q-functions
    qf1 = update_model(qf1, q1_grads)
    qf2 = update_model(qf2, q2_grads)

    q_loss = q1_loss + q2_loss

    # update value function
    vf = update_model(vf, v_grads)

    return (pi, qf1, qf2, vf, vf_target, af), (pi_loss, q_loss, v_loss, alpha_loss)
