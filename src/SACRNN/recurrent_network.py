"""
Networks for the Soft Actor-Critic in Jax Equinox
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jax.scipy.stats.norm import logpdf


class PolicyRNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    mu_layer: eqx.nn.Linear
    log_std_layer: list

    log_std_min: jnp.float32
    log_std_max: jnp.float32
    control_lim: jnp.float32
    hidden_size: int

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key,
        control_limit: float = 1,
        log_std_min: float = -20,
        log_std_max: float = 2,
        hidden_size: int = 2
    ):
        keys = jrandom.split(key, 3)
        self.hidden_size = hidden_size
        self.control_lim = control_limit

        # set log-std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # recurrent cell
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=keys[0])

        # mu layer
        self.mu_layer = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=keys[1])
        
        # set log-stdev layer
        self.log_std_layer = [eqx.nn.Linear(hidden_size, 32, key=keys[2]),
                                eqx.nn.Linear(32, out_size, key=keys[3])]
    
    @eqx.filter_jit
    def __call__(self, obs, control, key):
        input = jnp.hstack((obs, control))
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None
        
        x, _ = jax.lax.scan(f, hidden, input)

        # apply mu layer
        mu = jax.nn.tanh(self.mu_layer(x))

        # apply log-std layer
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
        
        return control * self.control_lim, log_prob
    
    def forward_step(self, obs, control, hidden, key):
        input = jnp.hstack((obs, control))

        h_out = self.cell(input, hidden)
        x = h_out

        # apply mu layer
        mu = jax.nn.tanh(self.mu_layer(x))

        # apply log-std layer
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
        
        return control * self.control_lim, log_prob, h_out
    
    def predict_step(self, obs, control, hidden):
        input = jnp.hstack((obs, control))

        hidden = self.cell(input, hidden)
        x = hidden

        # apply mu layer
        mu = jax.nn.tanh(self.mu_layer(x)) * self.control_lim

        return mu, hidden