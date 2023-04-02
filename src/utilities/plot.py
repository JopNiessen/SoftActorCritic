"""

"""

import numpy as np
import jax.numpy as jnp

from src.utilities.run_trial import run_trial


def training_performance(params, env, recurrent=False, hidden_size=0, **kwargs):
    T = kwargs.get('T', env.end_time)

    X, V = np.meshgrid(np.linspace(-3, 3, 3), np.linspace(-3, 3, 3))
    x0_space = np.column_stack((X.ravel(), V.ravel()))
    
    e_space = [int(s[1:]) for s in list(params.keys())]
    
    R = np.zeros(len(e_space))
    for it, ep in enumerate(e_space):
        if recurrent:
            controller = params[f'N{ep}']['pi'].predict_step
        else:
            controller = params[f'N{ep}']['pi'].predict
        for x0 in x0_space:
            _, _, _, rew = run_trial(env, controller, T=T, x0=jnp.array(x0), recurrent=recurrent, hidden_size=hidden_size)
            R[it] += rew
    return e_space, R/len(x0_space)

