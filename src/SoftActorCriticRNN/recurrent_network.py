"""
Networks for the Soft Actor-Critic in Jax Equinox
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jax.scipy.stats.norm import logpdf

from src.SoftActorCriticRNN.networks.GRU import GRUCell
from src.SoftActorCriticRNN.networks.WilsonCowan import WilsonCowanCell


class PolicyRNN(eqx.Module):
    EncoderCell: eqx.Module
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
        **kwargs
    ):
        keys = jrandom.split(key, 3)
        RNN_type = kwargs.get('RNN_type', 'WilsonCowan')
        self.hidden_size = kwargs.get('hidden_size', 8)
        self.control_limit = kwargs.get('control_limit', 1)

        # set log-std
        self.log_std_min = kwargs.get('log_std_min', -20)
        self.log_std_max = kwargs.get('log_std_max', 2)

        # set encoder
        if RNN_type == 'GRU':
            self.EncoderCell = GRUCell(in_size, self.hidden_size, key)
        elif RNN_type == 'WilsonCowan':
            self.EncoderCell = WilsonCowanCell(in_size, self.hidden_size, key)
        else:
            raise('RNN type not supported')

        # set mu layer
        self.C = eqx.nn.Linear(self.hidden_size, out_size, use_bias=False, key=keys[0])

        # set log-stdev layer
        self.log_std_layer = [eqx.nn.Linear(self.hidden_size, 32, key=keys[2]),
                                eqx.nn.Linear(32, out_size, key=keys[3])]
    
    def decoder(self, hidden):
        return self.C(hidden)

    @eqx.filter_jit
    def __call__(self, obs, control, key):
        input = jnp.hstack((obs, control))
        x = self.EncoderCell.forward_seq(input)

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
    
    @eqx.filter_jit
    def forward_step(self, obs, control, hidden, key):
        input = jnp.hstack((obs, control))

        hidden = self.EncoderCell(input, hidden)
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

        hidden = self.EncoderCell(input, hidden)

        # read out predicted control
        mu = jax.nn.tanh(self.decoder(hidden))

        return mu * self.control_limit, hidden


"""Q Network (critic)"""
class RQNetwork(eqx.Module):
    EncoderCell: eqx.Module
    general_layers: list

    hidden_size: jnp.int32
    
    def __init__(
        self,
        in_size: int,
        in2_size: int,
        width_size: int,
        depth: int,
        key: jrandom.PRNGKey,
        **kwargs
    ):
        """Initialize."""
        keys = jrandom.split(key, depth + 4)
        RNN_type = kwargs.get('RNN_type', 'WilsonCowan')
        self.hidden_size = kwargs.get('hidden_size', 32)
        
        """Encoder"""
        if RNN_type == 'GRU':
            self.EncoderCell = GRUCell(in_size, self.hidden_size, keys[0])
        elif RNN_type == 'WilsonCowan':
            self.EncoderCell = WilsonCowanCell(in_size, self.hidden_size, keys[0])
        else:
            raise('RNN type not supported')
        
        """Decoder"""
        self.general_layers = [eqx.nn.Linear(self.hidden_size+in2_size, width_size, key=keys[2])]
        for it in range(1, depth):
            self.general_layers += [eqx.nn.Linear(width_size, width_size, key=keys[2+it])]
        self.general_layers += [eqx.nn.Linear(width_size, 1, key=keys[-1])]

    @eqx.filter_jit
    def __call__(self, state_seq, control_seq, control):
        input = jnp.hstack((state_seq, control_seq))
        hidden = self.EncoderCell.forward_seq(input)

        return self.decoder(hidden, control)
    
    def decoder(self, hidden, control):
        x = jnp.hstack((hidden, control))
        for layer in self.general_layers[:-1]:
            x = jax.nn.relu(layer(x))
        q_val = self.general_layers[-1](x)
        return q_val


"""Value Network (critic)"""
class RValueNetwork(eqx.Module):
    EncoderCell: eqx.Module
    general_layers: list

    hidden_size: jnp.int32
    
    def __init__(
        self,
        in_size: int,
        width_size: int,
        depth: int,
        key: jrandom.PRNGKey,
        **kwargs
    ):
        """Initialize."""
        keys = jrandom.split(key, depth + 4)
        RNN_type = kwargs.get('RNN_type', 'WilsonCowan')
        self.hidden_size = kwargs.get('hidden_size', 32)

        """Encoder"""
        if RNN_type == 'GRU':
            self.EncoderCell = GRUCell(in_size, self.hidden_size, keys[0])
        elif RNN_type == 'WilsonCowan':
            self.EncoderCell = WilsonCowanCell(in_size, self.hidden_size, keys[0])
        else:
            raise('RNN type not supported')

        """Decoder"""
        self.general_layers = [eqx.nn.Linear(self.hidden_size, width_size, key=keys[2])]
        for it in range(1, depth):
            self.general_layers += [eqx.nn.Linear(width_size, width_size, key=keys[2+it])]
        self.general_layers += [eqx.nn.Linear(width_size, 1, key=keys[-1])]

    @eqx.filter_jit
    def __call__(self, state_seq, control_seq):
        input = jnp.hstack((state_seq, control_seq))
        hidden = self.EncoderCell.forward_seq(input)

        return self.decoder(hidden)
    
    def decoder(self, hidden):
        x = hidden
        for layer in self.general_layers[:-1]:
            x = jax.nn.relu(layer(x))
        q_val = self.general_layers[-1](x)
        return q_val



