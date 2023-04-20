"""

"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


class WilsonCowanCell(eqx.Module):
    A: eqx.Module
    B: eqx.Module

    hidden_size: int
    tau: float
    dt: float

    def __init__(self, in_size, hidden_size, key, use_bias=False, **kwargs):
        keys = jrandom.split(key, 3)
        self.hidden_size = hidden_size

        self.tau = kwargs.get('tau', 1)
        self.dt = kwargs.get('dt', 1)

        self.A = eqx.nn.Linear(hidden_size, hidden_size, use_bias=use_bias, key=keys[0])
        self.B = eqx.nn.Linear(in_size, hidden_size, use_bias=use_bias, key=keys[1])
    
    def __call__(self, input, hidden):
        return self.encoder(input, hidden)
    
    def forward_seq(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.encoder(inp, carry), None
        
        hidden, _ = jax.lax.scan(f, hidden, input)
        
        return hidden
    
    def derivative(self, u, r):
        return (1/self.tau) * (-r + self.A(jax.nn.tanh(r)) + self.B(u))

    def encoder(self, u, r):
        return r + self.dt * self.derivative(u, r)




