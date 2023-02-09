"""

"""

import numpy as np


def run(
    agent,
    T,
    x0 = None,
    n_obs = 2
):
    t_space = np.arange(0, T, agent.env.dt)

    Y = np.zeros((len(t_space), n_obs))
    S = np.zeros((len(t_space), n_obs))

    y = agent.env.reset(x0)
    for idx, _ in enumerate(np.arange(0, T, agent.env.dt)):
        u = agent.actor.model.predict(y)
        y, _, _, _ = agent.env.step(u)
        Y[idx] = y
        S[idx] = agent.env.state
    
    return S, Y