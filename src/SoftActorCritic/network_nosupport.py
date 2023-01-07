"""
Neural networks in jax equinox


"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax.scipy.stats.norm import logpdf

import equinox as eqx


class PolicyNetwork(eqx.Module):
    general_layers: list
    mu_layer: eqx.nn.Linear
    log_std_layer: eqx.nn.Linear

    log_std_min: jnp.float32
    log_std_max: jnp.float32
    control_lim: jnp.float32

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key,
        control_limit: float = 1,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        keys = jrandom.split(key, 4)
        self.control_lim = control_limit

        # set log-std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # set general layers
        self.general_layers = [eqx.nn.Linear(in_size, 128, key=keys[0]),
                                eqx.nn.Linear(128, 128, key=keys[1])]

        # set log_std layer
        self.log_std_layer = eqx.nn.Linear(128, out_size, key=keys[2])

        # set mu layer
        self.mu_layer = eqx.nn.Linear(128, out_size, key=keys[3])
    
    def __call__(self, state, key, deterministic=False):
        x = state
        for layer in self.general_layers:
            x = jax.nn.relu(layer(x))
        
        # apply mu layer
        mu = jax.nn.tanh(self.mu_layer(x))

        if deterministic:
            return mu * self.control_lim, None
        else:
            # apply log-std layer
            log_std = jnp.tanh(self.log_std_layer(x))
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            std = jnp.exp(log_std)
            
            # sample control
            z = mu + std * jrandom.normal(key, (1,))
            
            # squeez and normalize
            control = jnp.tanh(z)                       # TODO: jnp.tanh appearantly can yield a value slightly higher than 1.
            log_prob = logpdf(z, loc=mu, scale=std) - jnp.log(1 - control**2 + 1e-5)
            
            return control * self.control_lim, log_prob


"""Q Network"""
class QNetwork(eqx.Module):
    hidden1: eqx.nn.Linear
    hidden2: eqx.nn.Linear
    out: eqx.nn.Linear
    
    def __init__(self, in_dim, key):
        """Initialize."""
        keys = jrandom.split(key, 3)
        self.hidden1 = eqx.nn.Linear(in_dim, 128, key=keys[0])
        self.hidden2 = eqx.nn.Linear(128, 128, key=keys[1])
        self.out = eqx.nn.Linear(128, 1, key=keys[2])

    def __call__(self, state, control):
        """Forward method implementation."""
        x = jnp.hstack((state, control))
        x = jax.nn.relu(self.hidden1(x))
        x = jax.nn.relu(self.hidden2(x))
        value = self.out(x)
        
        return value


"""Value Network"""
class ValueNetwork(eqx.Module):
    hidden1: jnp.ndarray
    hidden2: jnp.ndarray
    out: jnp.ndarray

    def __init__(self, in_dim, key):
        """Initialize."""
        keys = jrandom.split(key, 3)
        self.hidden1 = eqx.nn.Linear(in_dim, 128, key=keys[0])
        self.hidden2 = eqx.nn.Linear(128, 128, key=keys[1])
        self.out = eqx.nn.Linear(128, 1, key=keys[2])

    def __call__(self, state):
        """Forward method implementation."""
        x = jax.nn.relu(self.hidden1(state))
        x = jax.nn.relu(self.hidden2(x))
        value = self.out(x)
        
        return value

