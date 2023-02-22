"""
Networks for the Soft Actor-Critic in Jax Equinox
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jax.scipy.stats.norm import logpdf


"""Policy Network (actor)"""
class LinearPolicyNetwork(eqx.Module):
    mu_layer: eqx.nn.Linear
    log_std_layer: list

    log_std_min: jnp.float32
    log_std_max: jnp.float32
    control_scale: jnp.float32

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key,
        control_scale: float = 1,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        keys = jrandom.split(key, 3)
        self.control_scale = control_scale

        # set log-std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # set mu layer
        self.mu_layer = eqx.nn.Linear(in_size, out_size, use_bias=False, key=keys[0])

        # set log_std layer
        self.log_std_layer = [eqx.nn.Linear(in_size, 32, key=keys[1]),
                                eqx.nn.Linear(32, out_size, key=keys[2])]
    
    @jax.jit
    def __call__(self, state, key):
        x = state
        
        # apply mu layer
        mu = jax.nn.tanh(self.mu_layer(x))

        for layer in self.log_std_layer[:-1]:
            x = jax.nn.relu(layer(x))
        log_std = jnp.tanh(self.log_std_layer[-1](x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = jnp.exp(log_std)
        
        # sample control
        z = mu + std * jrandom.normal(key, (1,))
        
        # squeez and normalize
        control = jnp.tanh(z)                       # TODO: jnp.tanh appearantly can yield a value slightly higher than 1.
        log_prob = logpdf(z, loc=mu, scale=std) - jnp.log(1 - control**2 + 1e-5)
        
        return control * self.control_scale, log_prob
    
    def predict(self, state):
        mu = jax.nn.tanh(self.mu_layer(state))
        return mu * self.control_scale


"""Q Network (critic)"""
class QNetwork(eqx.Module):
    general_layers: list
    
    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, key: jrandom.PRNGKey):
        """Initialize."""
        keys = jrandom.split(key, depth + 2)
        self.general_layers = [eqx.nn.Linear(in_size, width_size, key=keys[0])]
        for it in range(1, depth):
            self.general_layers += [eqx.nn.Linear(width_size, width_size, key=keys[it])]
        self.general_layers += [eqx.nn.Linear(width_size, out_size, key=keys[-1])]

    @jax.jit
    def __call__(self, state, control):
        """Forward method implementation."""
        x = jnp.hstack((state, control))
        #x = jax.nn.relu(jax.tree_multimap(lambda layer, x: layer(x), self.general_layers[:-1], x))
        for layer in self.general_layers[:-1]:
            x = jax.nn.relu(layer(x))
        q_value = self.general_layers[-1](x)
        return q_value


"""Value Network (critic)"""
class ValueNetwork(eqx.Module):
    general_layers: list
    
    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, key: jrandom.PRNGKey):
        """Initialize."""
        keys = jrandom.split(key, depth + 2)
        self.general_layers = [eqx.nn.Linear(in_size, width_size, key=keys[0])]
        for it in range(1, depth):
            self.general_layers += [eqx.nn.Linear(width_size, width_size, key=keys[it])]
        self.general_layers += [eqx.nn.Linear(width_size, out_size, key=keys[-1])]

    @jax.jit
    def __call__(self, state):
        """Forward method implementation."""
        x = state
        for layer in self.general_layers[:-1]:
            x = jax.nn.relu(layer(x))
        value = self.general_layers[-1](x)
        return value
