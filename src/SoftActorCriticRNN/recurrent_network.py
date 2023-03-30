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
    A: eqx.Module
    B: eqx.Module
    C: eqx.Module
    log_std_layer: list

    log_std_min: jnp.float32
    log_std_max: jnp.float32
    hidden_size: int
    control_limit: jnp.float32

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key,
        log_std_min: float = -20,
        log_std_max: float = 2,
        hidden_size: int = 2,
        control_limit: float = 1
    ):
        keys = jrandom.split(key, 3)
        self.hidden_size = hidden_size

        self.control_limit = control_limit

        # set log-std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Wilson-Cowan model
        self.A = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[0])
        self.B = eqx.nn.Linear(in_size, hidden_size, use_bias=False, key=keys[1])
        self.C = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=keys[0])

        # set log-stdev layer
        self.log_std_layer = [eqx.nn.Linear(hidden_size, 32, key=keys[2]),
                                eqx.nn.Linear(32, out_size, key=keys[3])]
    
    def encoder(self, input, hidden):
        return jax.nn.tanh(self.A(hidden) + self.B(input))
    
    def decoder(self, hidden):
        return self.C(hidden)

    @eqx.filter_jit
    def __call__(self, obs, control, key):
        input = jnp.hstack((obs, control))
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.encoder(inp, carry), None
        
        x, _ = jax.lax.scan(f, hidden, input)

        # apply mu layer
        mu = jax.nn.tanh(self.decoder(x))

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
        
        return control * self.control_limit, log_prob
    
    def forward_step(self, obs, control, hidden, key):
        input = jnp.hstack((obs, control))

        hidden = self.encoder(input, hidden)
        x = hidden

        # apply mu layer
        mu = jax.nn.tanh(self.decoder(x))

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
        
        return control * self.control_limit, log_prob, hidden
    
    def predict_step(self, obs, control, hidden):
        input = jnp.hstack((obs, control))

        hidden = self.encoder(input, hidden)

        # read out predicted control
        mu = jax.nn.tanh(self.decoder(hidden))

        return mu * self.control_limit, hidden


"""Q Network (critic)"""
class RQNetwork(eqx.Module):
    general_layers: list
    A: eqx.Module
    B: eqx.Module

    hidden_size: jnp.int32
    
    def __init__(
        self,
        obs_size: int,
        control_size: int,
        width_size: int,
        depth: int,
        key: jrandom.PRNGKey,
        hidden_size: int = 8
    ):
        """Initialize."""
        keys = jrandom.split(key, depth + 4)
        self.hidden_size = hidden_size

        """Wilson-Cowan model"""
        self.A = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[0])
        self.B = eqx.nn.Linear(obs_size + control_size, hidden_size, use_bias=False, key=keys[1])

        """Decoder"""
        self.general_layers = [eqx.nn.Linear(hidden_size+control_size, width_size, key=keys[2])]
        for it in range(1, depth):
            self.general_layers += [eqx.nn.Linear(width_size, width_size, key=keys[2+it])]
        self.general_layers += [eqx.nn.Linear(width_size, 1, key=keys[-1])]

    @eqx.filter_jit
    def __call__(self, state_seq, control_seq, control):
        input = jnp.hstack((state_seq, control_seq))
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.encoder(inp, carry), None
        
        hid, _ = jax.lax.scan(f, hidden, input)

        q_value = self.decoder(hid, control)
        return q_value
    
    def encoder(self, input, hidden):
        return jax.nn.sigmoid(self.A(hidden) + self.B(input))

    def decoder(self, hidden, control):
        x = jnp.hstack((hidden, control))
        for layer in self.general_layers[:-1]:
            x = jax.nn.relu(layer(x))
        q_val = self.general_layers[-1](x)
        return q_val


"""Value Network (critic)"""
class RValueNetwork(eqx.Module):
    general_layers: list
    A: eqx.Module
    B: eqx.Module

    hidden_size: jnp.int32
    
    def __init__(
        self,
        obs_size: int,
        control_size: int,
        width_size: int,
        depth: int,
        key: jrandom.PRNGKey,
        hidden_size: int = 8
    ):
        """Initialize."""
        keys = jrandom.split(key, depth + 4)
        self.hidden_size = hidden_size

        """Wilson-Cowan model"""
        self.A = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[0])
        self.B = eqx.nn.Linear(obs_size+control_size, hidden_size, use_bias=False, key=keys[1])

        """Decoder"""
        self.general_layers = [eqx.nn.Linear(hidden_size, width_size, key=keys[2])]
        for it in range(1, depth):
            self.general_layers += [eqx.nn.Linear(width_size, width_size, key=keys[2+it])]
        self.general_layers += [eqx.nn.Linear(width_size, 1, key=keys[-1])]

    @eqx.filter_jit
    def __call__(self, state_seq, control_seq):
        """Forward method implementation."""
        input = jnp.hstack((state_seq, control_seq))
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.encoder(inp, carry), None
        
        hid, _ = jax.lax.scan(f, hidden, input)

        value = self.decoder(hid)
        return value
    
    def encoder(self, input, hidden):
        return jax.nn.relu(self.A(hidden) + self.B(input))

    def decoder(self, hidden):
        x = hidden
        for layer in self.general_layers[:-1]:
            x = jax.nn.relu(layer(x))
        q_val = self.general_layers[-1](x)
        return q_val



