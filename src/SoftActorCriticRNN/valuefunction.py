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
from src.SoftActorCriticRNN.recurrent_network import RValueNetwork


class RValueFunction:
    """
    Value function (critic) of the Soft Actor-Critic (SAC) algorithm
    """
    def __init__(self, obs_size, control_size, learning_rate, key, **kwargs):
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
        depth = kwargs.get('depth', 1)
        hidden_size = kwargs.get('hidden_size', 32)
        self.model = RValueNetwork(obs_size, control_size, width, depth, key, hidden_size=hidden_size)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model)
    
    def __call__(self, state_seq, control_seq) -> jnp.ndarray:
        """
        Evaluate the value function for a given state
        :param state: state [jnp.ndarray]
        :return: value [jnp.ndarray]
        """
        return self.model(state_seq, control_seq)
    
    @eqx.filter_jit
    def loss(self, model, state_seq, control_seq, target) -> jnp.ndarray:
        """
        Compute loss of the value-network
        :param model: equinox network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        """
        pred = jax.vmap(model)(state_seq, control_seq)
        #pred = jnp.reshape(pred, target.shape) #TODO: may be redundant > check
        return jnp.mean((pred - target)**2)
    
    #@eqx.filter_jit
    def value_and_grad(self, state_seq, control_seq, target):
        """
        Compute loss and gradient of the value-network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        :return: gradients
        """
        return eqx.filter_value_and_grad(self.loss)(self.model, state_seq, control_seq, target)
    
    def update(self, grads):
        """
        Update network weights
        :param grads: gradients
        """
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)