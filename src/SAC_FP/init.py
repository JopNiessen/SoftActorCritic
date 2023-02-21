"""
SAC network initialization functions
"""

# import libraries
import jax.numpy as jnp
import optax

# import local libraries
from src.SAC_FP.network import LinearPolicyNetwork, QNetwork, ValueNetwork


"""Initialize pi-function"""
def pi_init(in_size, out_size, lr, key):
    model = LinearPolicyNetwork(in_size, out_size, key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model)

    return model, optimizer, opt_state


"""Initialize value function"""
def vf_init(in_size, out_size, width, depth, lr, key):
    model = ValueNetwork(in_size, out_size, width, depth, key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model)

    return model, optimizer, opt_state


"""Initialize q-function"""
def qf_init(in_size, out_size, width, depth, lr, key):
    model = QNetwork(in_size, out_size, width, depth, key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model)

    return model, optimizer, opt_state


"""Initialize entropy term"""
def alpha_init(lr):
    alpha_target = -1
    log_alpha = jnp.array([0], dtype=float)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_alpha)
    return log_alpha, optimizer, opt_state, alpha_target
