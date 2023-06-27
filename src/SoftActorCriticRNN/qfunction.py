"""
Q-function (critic) of the Soft Actor-Critic (SAC) algorithm
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
from typing import Tuple

# import local libraries
from src.SoftActorCriticRNN.recurrent_network import RQNetwork
from src.SoftActorCriticRNN.utilities.functional import clip_grads


class RQFunction:
    """
    Q-function (critic) of the Soft Actor-Critic (SAC) algorithm
    """
    def __init__(self, in_size, in2_size, learning_rate, key, **kwargs):
        """
        Initialize the Q-function
        :param in_size: input size [int]
        :param learning_rate: learning rate [float]
        :param key: key [jrandom.PRNGKey]
        :param out_size: output size (default: 1) [int]
        :param width: network width (default: 32) [int]
        :param depth: network depth (default: 2) [int]
        """
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 1)
        self.model = RQNetwork(in_size, in2_size, width, depth, key, **kwargs)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.model)
    
    def __call__(self, state_seq, control_seq, control):
        """
        Evaluate the Q-function for a given state and control
        :param state: state [jnp.ndarray]
        :param control: control [jnp.ndarray]
        :return: Q-value [jnp.ndarray]
        """
        return self.model(state_seq, control_seq, control)
    
    @eqx.filter_jit
    def loss(self, model, state_seq, control_seq, control, target) -> jnp.ndarray:
        """
        Compute loss of the Q-network
        :param model: equinox network
        :param state: state [jnp.ndarray]
        :param control: control [jnp.ndarray]
        :param target: target Q-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        """
        pred = jax.vmap(model)(state_seq, control_seq, control)
        return jnp.mean((pred - target)**2)
    
    #@eqx.filter_jit
    def value_and_grad(self, state_seq, control_seq, control, target) -> Tuple:
        """
        Compute loss and gradient of the Q-network
        :param state: state [jnp.ndarray]
        :param control: control [jnp.ndarray]
        :param target: target Q-value [jnp.ndarray]
        :return: loss [jnp.ndarray]
        :return: gradients
        """
        return eqx.filter_value_and_grad(self.loss)(self.model, state_seq, control_seq, control, target)
    
    def update(self, grads):
        """
        Update network weights
        :param grads: gradients
        """
        grads = clip_grads(grads)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)


"""Function based approach"""
def generate_instance(in_size, in2_size, learning_rate, key, **kwargs):
    """
    Initialize the Q-function
    :param in_size: input size [int]
    :param learning_rate: learning rate [float]
    :param key: key [jrandom.PRNGKey]
    :param out_size: output size (default: 1) [int]
    :param width: network width (default: 32) [int]
    :param depth: network depth (default: 2) [int]
    """
    width = kwargs.get('width', 64)
    depth = kwargs.get('depth', 1)
    model = RQNetwork(in_size, in2_size, width, depth, key, **kwargs)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)
    return model, optimizer, opt_state


@eqx.filter_value_and_grad
def loss_fn(model, state_seq, control_seq, control, target):
    """
    Compute loss of the Q-network
    :param model: equinox network
    :param state: state [jnp.ndarray]
    :param control: control [jnp.ndarray]
    :param target: target Q-value [jnp.ndarray]
    :return: loss [jnp.ndarray]
    """
    pred = jax.vmap(model)(state_seq, control_seq, control)
    return jnp.mean((pred - target)**2)


@eqx.filter_jit
def make_step(params, state_seq, control_seq, control, target):
    (model, optim, opt_state) = params
    loss, grads = loss_fn(model, state_seq, control_seq, control, target)
    grads = clip_grads(grads)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, (model, optim, opt_state)

