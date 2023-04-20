"""

"""

import jax
import jax.numpy as jnp
import equinox as eqx


class GRUCell(eqx.Module):
    cell: eqx.Module
    hidden_size: int

    def __init__(self, in_size, hidden_size, key, use_bias=False, **kwargs):
        self.hidden_size = hidden_size

        self.cell = eqx.nn.GRUCell(in_size, hidden_size, use_bias=use_bias, key=key)
    
    def __call__(self, input, hidden):
        return self.encoder(input, hidden)
    
    def forward_seq(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.encoder(inp, carry), None
        
        hidden, _ = jax.lax.scan(f, hidden, input)
        
        return hidden

    def encoder(self, u, r):
        return self.cell(u, r)
