"""
2-Dimensional Linear Quadratic (LQ) system with gaussian noise

by J. Niessen
created on: 2022.10.24
"""


import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from numpy.random import randint


# For now, it uses time steps (dt) of 1 sec. With x1 = 'velocity in (m/s)'
class StochasticDoubleIntegrator:
    def __init__(self, x0, b=1, k=.02, dt=.1, end_time=4):
        """
        This class describes a 2 dimensional linear dynamical system with gaussian noise

        :param x0: initial state
        :param b: bias term
        :param k: friction
        :param dt: time step size
        :param time_horizon: end time
        """
        self.state = x0
        self.done = False

        self.t = 0
        self.dt = dt
        self.end_time = end_time
        
        self.dim = len(x0)
        self.boundary = jnp.array([jnp.inf, jnp.inf])

        self.observation_space = x0
        self.action_space = jnp.array([[-1, 1]])*2

        """System parameters"""
        self.A = jnp.array([[0, 1], [0, -k]])
        self.B = jnp.array([0, b])
        self.C = jnp.identity(self.dim)
        self.v = jnp.array([[0, 0], [0, 0]])
        self.w = jnp.array([[0, 0], [0, .5]])

        """Cost parameters"""
        self.F = jnp.array([[1, 0], [0, 0]])
        self.G = jnp.array([[1, 0], [0, 0]])
        self.R = 1
    
    def step(self, control, key=None):
        key, subkey = jrandom.split(random_key(key))
        xi = jrandom.normal(key, (self.dim, ))
        self.state = self.state + self.dt * (jnp.dot(self.A, self.state) + self.B * control) + jnp.sqrt(self.dt) * np.dot(self.w, xi)
        self.t += 1
        
        observation = self._get_obs(subkey)
        reward = -self.cost(self.state, control)[0]

        self._check_boundary()

        return observation, reward, self.done, {}

    def _check_boundary(self):
        if any(self.state >= self.boundary):
            self.done = True
        elif self.t >= self.end_time:
            self.done = True

    def _get_obs(self, key):
        """
        Observe the state (x) according to: y(n) = Cx(n) + Vxi
        :return: state observation (y)
        """
        xi = jrandom.normal(key, (self.dim, ))
        return np.dot(self.C, self.state) + np.dot(self.v, xi)

    def cost(self, state, control):
        """
        (Marginal) cost
        :param x: state
        :param u: control
        :return: marginal cost
        """
        x, u = state, control
        return (x.T @ self.G @ x + self.R * u**2)*self.dt

    def terminal_cost(self, state):
        """
        Cost in final timestep (t=T)
        :param x: state
        :return: end cost
        """
        x = state
        return x.T @ self.F @ x

    def reset(self, key=jrandom.PRNGKey(randint(0, high=1000))):
        """
        Reset state
        :param x0: initial state
        """
        key = random_key(key)
        self.state = jrandom.normal(key, (self.dim,))
        self.t = 0
        self.done = False
        return self.state
    
    def sample(self, key=None):
        key = random_key(key)
        return jrandom.uniform(key, (1,), minval=self.action_space[0,0], maxval=self.action_space[0,1])
    
    def close(self):
        pass
    


def random_key(key, high=1000):
    if key == None:
        return jrandom.PRNGKey(randint(0, high=high))
    else:
        return key



