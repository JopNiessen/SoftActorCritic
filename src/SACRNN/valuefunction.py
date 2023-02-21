"""
Value function (critic) of the Soft Actor-Critic (SAC) algorithm
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

# import local libraries
from src.SACRNN.network import ValueNetwork


class ValueFunction:
    """
    Value function (critic) of the Soft Actor-Critic (SAC) algorithm
    """
    def __init__(self, in_size: int, learning_rate: float, key: jrandom.PRNGKey, **kwargs):
        """
        Initialize the value function
        :param in_size: input size [int]
        :param learning_rate: learning rate [float]
        :param key: key [jrandom.PRNGKey]
        :param out_size: output size (default: 1) [int]
        :param width: network width (default: 32) [int]
        :param depth: network depth (default: 2) [int]
        """
        out_size = kwargs.get('out_size', 1)
        width = kwargs.get('width', 32)
        depth = kwargs.get('depth', 2)
        self.model = ValueNetwork(in_size, out_size, width, depth, key)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model)
    
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the value function for a given state
        :param state: state [jnp.ndarray]
        :return: value [jnp.ndarray]
        """
        return self.model(state)
    
    def loss(self, model, state, target) -> jnp.ndarray:
        """
        Compute loss of the value-network
        :param model: equinox network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        """
        pred = jax.vmap(model)(state)
        return jnp.mean((pred - target)**2)
    
    #@eqx.filter_jit
    def value_and_grad(self, state, target):
        """
        Compute loss and gradient of the value-network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        :return: gradients
        """
        return eqx.filter_value_and_grad(self.loss)(self.model, state, target)
    
    #@eqx.filter_jit
    def update(self, grads):
        """
        Update network weights
        :param grads: gradients
        """
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)

