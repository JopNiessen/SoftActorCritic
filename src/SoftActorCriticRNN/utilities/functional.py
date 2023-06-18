

import jax
import jax.numpy as jnp


def clip_grads(grads, clip_value=1):
    return jax.tree_map(lambda x: jnp.clip(x, -clip_value, clip_value), grads)
