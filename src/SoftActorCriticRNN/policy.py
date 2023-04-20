"""
Gaussian policy (actor) of the Soft Actor-Critic (SAC) algorithm
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
from typing import Callable, Tuple

# import local libraries
from src.SoftActorCriticRNN.recurrent_network import PolicyRNN

class PolicyFunctionRNN:
    """
    Gaussian policy (actor) of the Soft Actor-Critic (SAC) algorithm
    """
    def __init__(self, in_size: int, out_size: int, learning_rate: float, key: jrandom.PRNGKey, **kwargs):
        """
        Initialize the policy function
        :param in_size: input size [int]
        :param out_size: output size [int]
        :param learning_rate: learning rate [float]
        :param key: key [jrandom.PRNGKey]
        :param control_limit: maximal control magnitude [int]
        """
        self.model = PolicyRNN(in_size, out_size, key, **kwargs)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model)
    
    def __call__(self, state_traj: jnp.ndarray, control_traj: jnp.ndarray, key: jnp.ndarray):
        """
        Evaluate the policy for a given state
        :param state: state [jnp.ndarray]
        :param key: PRNGKey [jnp.ndarray]
        :return: control [jnp.ndarray]
        """
        return self.model(state_traj, control_traj, key)
    
    #@eqx.filter_jit
    def loss(self, model, state_seq, control_seq, state, alpha, v_pred, qf1, qf2, keys) -> jnp.ndarray:
        """
        Compute loss of the value-network
        :param model: equinox network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        """
        new_control, log_prob = jax.vmap(model)(state_seq, control_seq, keys)
        q1 = jax.vmap(qf1)(state_seq, control_seq, new_control)
        q2 = jax.vmap(qf2)(state_seq, control_seq, new_control)
        q_pred = jax.vmap(jax.lax.min)(q1, q2)
        advantage = q_pred - v_pred
        return jnp.mean(alpha * log_prob - advantage)
    
    #@eqx.filter_jit
    def value_and_grad(self, state_traj, control_traj, state, alpha, v_pred, qf1, qf2, keys) -> Tuple:
        """
        Compute loss and gradient of the policy-network
        :param state: state [jnp.ndarray]
        :param alpha: entropy [jnp.ndarray]
        :param v_pred: value [jnp.ndarray]
        :param q_min: Q-function [Callable]
        :param keys: PRNGKey [jnp.ndarray]
        :return: loss [jnp.ndarray]
        :return: gradients
        """
        return eqx.filter_value_and_grad(self.loss)(self.model, state_traj, control_traj, state, alpha, v_pred, qf1, qf2, keys)
    
    def update(self, grads):
        """
        Update network weights
        :param grads: gradients
        """
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)


