"""

"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def run_trial(
    env,
    controller,
    key,
    x0 = None,
    **kwargs
):
    recurrent = kwargs.get('recurrent', False)
    hidden_size = kwargs.get('hidden_size', 1)
    T = kwargs.get('T', env.end_time)

    t_space = np.arange(0, T, env.dt)

    y = env.reset(x0)
    s = env.state
    u = 0
    hid = jnp.zeros(hidden_size)

    hist_size = kwargs.get('history_size', 0)
    window = kwargs.get('window', False)
    state_memory = np.zeros((hist_size, len(y)))
    control_memory = np.zeros((hist_size, 1))

    Y = np.zeros((len(t_space)+1, len(y)))
    S = np.zeros((len(t_space)+1, len(s)))
    Y[0] = y
    S[0] = s
    U = np.zeros((len(t_space), 1))
    R = 0

    for it in range(len(t_space)):
        key, subkey = jrandom.split(key)
        if recurrent:
            if window:
                u, hid = controller(state_memory, control_memory, subkey)
            else:
                u, hid = controller(y, u, hid)
        else:
            u = controller(y)
        y, rew, _, _ = env.step(u, key=key)
        if window:
            state_memory[1:] = state_memory[:-1]
            state_memory[-1] = y
            control_memory[1:] = control_memory[:-1]
            control_memory[-1] = u
        Y[it+1] = y
        S[it+1] = env.state
        U[it] = u
        R += rew
    
    return S, Y, U, R

