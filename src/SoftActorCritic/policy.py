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
from src.SoftActorCritic.network import PolicyNetwork


class PolicyFunction:
    """
    Gaussian policy (actor) of the Soft Actor-Critic (SAC) algorithm
    """
    def __init__(self, in_size: int, out_size: int, learning_rate: float, key: jrandom.PRNGKey, control_limit: int = 1):
        """
        Initialize the policy function
        :param in_size: input size [int]
        :param out_size: output size [int]
        :param learning_rate: learning rate [float]
        :param key: key [jrandom.PRNGKey]
        :param control_limit: maximal control magnitude [int]
        """
        self.model = PolicyNetwork(in_size, out_size, key, control_limit=control_limit)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model)
    
    def __call__(self, state: jnp.ndarray, key: jnp.ndarray, deterministic: bool = False):
        """
        Evaluate the policy for a given state
        :param state: state [jnp.ndarray]
        :param key: PRNGKey [jnp.ndarray]
        :param deterministic: indicate if policy is deterministic [bool]
        :return: control [jnp.ndarray]
        """
        return self.model(state, key, deterministic=deterministic)
    
    #@eqx.filter_jit
    def loss(self, model, state, alpha, v_pred, q_min, keys) -> jnp.ndarray:
        """
        Compute loss of the value-network
        :param model: equinox network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        """
        action, log_prob = jax.vmap(model)(state, keys)
        q_pred = jax.vmap(q_min)(state, action)
        q_pred = jnp.reshape(q_pred, v_pred.shape)
        advantage = q_pred - v_pred
        advantage = jnp.reshape(advantage, log_prob.shape)
        return jnp.mean(alpha * log_prob - advantage)
    
    def value_and_grad(self, state: jnp.ndarray, alpha: jnp.ndarray, v_pred: jnp.ndarray, q_min: Callable, keys: jnp.ndarray) -> Tuple:
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
        return eqx.filter_value_and_grad(self.loss)(self.model, state, alpha, v_pred, q_min, keys)
    
    def update(self, grads):
        """
        Update network weights
        :param grads: gradients
        """
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)

