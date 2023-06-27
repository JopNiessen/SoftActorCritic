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
from src.SoftActorCriticRNN.utilities.functional import clip_grads


class RValueFunction:
    """
    Value function (critic) of the Soft Actor-Critic (SAC) algorithm
    """
    def __init__(self, in_size, learning_rate, key, **kwargs):
        """
        Initialize the value function
        :param in_size: input size [int]
        :param learning_rate: learning rate [float]
        :param key: key [jrandom.PRNGKey]
        :param out_size: output size (default: 1) [int]
        :param width: network width (default: 32) [int]
        :param depth: network depth (default: 2) [int]
        """
        width = kwargs.get('width', 32)
        depth = kwargs.get('depth', 1)
        self.model = RValueNetwork(in_size, width, depth, key, **kwargs)
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
    def loss(self, model, state_seq, control_seq, target):
        """
        Compute loss of the value-network
        :param model: equinox network
        :param state: state [jnp.ndarray]
        :param target: target-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        """
        pred = jax.vmap(model)(state_seq, control_seq)
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
        grads = clip_grads(grads)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)


"""Function based approach"""
def generate_instance(in_size, learning_rate, key, **kwargs):
    """
    Generate value function
    :param in_size: input size [int]
    :param learning_rate: learning rate [float]
    :param key: key [jrandom.PRNGKey]
    :param out_size: output size (default: 1) [int]
    :param width: network width (default: 32) [int]
    :param depth: network depth (default: 2) [int]
    :return: value function [RValueFunction]
    """
    width = kwargs.get('width', 32)
    depth = kwargs.get('depth', 1)
    model = RValueNetwork(in_size, width, depth, key, **kwargs)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)
    return (model, optimizer, opt_state)


@eqx.filter_value_and_grad
def loss_fn(model, state_seq, control_seq, target):
    """
    Compute loss of the value-network
    :param model: equinox network
    :param state: state [jnp.ndarray]
    :param target: target-value [jnp.ndarray]
    :return: loss [jnp.ndarray]
    """
    pred = jax.vmap(model)(state_seq, control_seq)
    return jnp.mean((pred - target)**2)


@eqx.filter_jit
def make_step(params, state_seq, control_seq, target):
    (model, optim, opt_state) = params
    loss, grads = loss_fn(model, state_seq, control_seq, target)
    grads = clip_grads(grads)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, (model, optim, opt_state)
