"""
Directly optimize designed functions
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

class Quadratic(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size, key):
        #keys = jrandom.split(key)
        self.weight = -jnp.array([[1, .5], [.5, 1]])
        self.bias = jnp.array([0])
        #self.weight = jrandom.normal(keys[0], (in_size, in_size))
        #self.bias = jrandom.normal(keys[1], (1,))
    
    def __call__(self, input):
        x = input
        return x.T @ self.weight @ x + self.bias


class QuadraticFunction(eqx.Module):
    general_layers: list

    def __init__(self, in_size, key):
        self.general_layers = [Quadratic(in_size, key)]
    
    def __call__(self, input):
        x = input
        for layer in self.general_layers:
            x = layer(x)
        return x


class Linear(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, in_size, key):
        keys = jrandom.split(key)
        self.b = jrandom.normal(keys[0], (1,))
        self.W = jrandom.normal(keys[1], (in_size, 1))
    
    def __call__(self, input):
        x = jnp.dot(self.W, input) + self.b
        return x

