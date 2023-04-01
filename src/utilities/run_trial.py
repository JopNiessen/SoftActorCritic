"""

"""

import jax.numpy as jnp
import numpy as np


def run_trial(
    env,
    controller,
    x0 = None,
    **kwargs
):
    recurrent = kwargs.get('recurrent', False)
    hidden_size = kwargs.get('hidden_size', 0)
    T = kwargs.get('T', env.end_time)

    t_space = np.arange(0, T, env.dt)

    y = env.reset(x0)
    s = env.state
    u = 0
    hid = jnp.zeros(hidden_size)

    Y = np.zeros((len(t_space), len(y)))
    S = np.zeros((len(t_space), len(s)))
    U = np.zeros((len(t_space), 1))

    for it in range(len(t_space)):
        if recurrent:
            u, hid = controller(y, u, hid)
        else:
            u = controller(y)
        y, _, _, _ = env.step(u)
        Y[it] = y
        S[it] = env.state
        U[it] = u
    
    return S, Y, U

