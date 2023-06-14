"""
Run functions for optimal control algorithms.
"""

import numpy as np


def run_LQR(env, controller, T=20, dt=.1, x0=None):
    """
    Run LQR controller on environment.
    :param env: environment
    :param controller: LQR controller
    :param T: time horizon
    :param dt: time step
    :param x0: initial state
    :return: state, observation, action, reward
    """
    time_space = np.arange(0, T, dt)
    X = np.zeros((len(time_space)+1, env.dim))
    Y = np.zeros((len(time_space)+1, env.dim))
    U = np.zeros((len(time_space), env.dim))
    R = np.zeros((len(time_space), env.dim))

    y = env.reset(x0=x0)

    X[0] = env.state
    Y[0] = y

    for it, _ in enumerate(time_space):
        u = controller(env.state)
        y, rew, _, _ = env.step(u)

        X[it+1] = env.state
        Y[it+1] = y
        U[it] = u
        R[it] = rew
    
    return X, Y, U, R


def run_LQG(env, controller, T=20, dt=.1, x0=None):
    """
    Run LQG controller on environment.
    :param env: environment
    :param controller: LQG controller
    :param T: time horizon
    :param dt: time step
    :param x0: initial state
    :return: state, observation, action, reward
    """
    time_space = np.arange(0, T, dt)
    X = np.zeros((len(time_space)+1, env.dim))
    Y = np.zeros((len(time_space)+1, env.dim))
    U = np.zeros((len(time_space), env.dim))
    R = np.zeros((len(time_space), env.dim))

    y = env.reset(x0=x0)
    controller.x_est = y
    u = None

    X[0] = env.state
    Y[0] = y

    for it, _ in enumerate(time_space):
        u = controller.step(y, u, dt=dt)
        y, rew, _, _ = env.step(u[1])

        X[it+1] = env.state
        Y[it+1] = y
        U[it] = u
        R[it] = rew
    
    return X, Y, U, R