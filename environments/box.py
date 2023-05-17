"""
2-Dimensional Linear Quadratic (LQ) system with gaussian noise

by J. Niessen
created on: 2022.10.24
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from numpy.random import randint


class Box_SDI:
    def __init__(self, b=1, k=.2, dt=.1, end_time=20, **kwargs):
        """
        This class describes a 2 dimensional linear dynamical system with gaussian noise

        :param b: bias term [float]
        :param k: friction constant [float]
        :param dt: time step size [float]
        :param end_time: end-time of a trial [int]
        """
        self.state = None
        self.done = False
        self.edge = False

        self.t = 0
        self.dt = dt
        self.end_time = end_time

        self.random_bias = kwargs.get('random_bias', False)
        
        self.dim = 2
        self.min = kwargs.get('min', jnp.array([-5, -jnp.inf]))
        self.max = kwargs.get('max', jnp.array([5, jnp.inf]))

        """System parameters"""
        self.A = jnp.array([[0, 1], [0, -k]])
        self.B = jnp.array([0, b])
        self.C = jnp.identity(self.dim)
        self.v = jnp.array([[.5, 0], [0, .5]])      # observation noise
        self.w = jnp.array([[0, 0], [0, .2]])       # system noise

        """Cost parameters"""
        self.F = jnp.array([[1, 0], [0, 0]])
        self.G = jnp.array([[1, 0], [0, 0]])
        self.R = 1

        self.reset()
    
    def predict_deriv(self, state, control):
        """
        State derivative
        :param state: state [jnp.array]
        :param control: control [float]
        """
        return jnp.dot(self.A, state) + self.B * control
    
    def step(self, control, key=None):
        key, subkey = jrandom.split(random_key(key))
        xi = jrandom.normal(key, (self.dim, ))
        self.state = self.state + self.dt * (jnp.dot(self.A, self.state) + self.B * control) + np.sqrt(self.dt) * np.dot(self.w, xi)
        self.t += self.dt
        
        observation = self._get_obs(subkey)
        reward = -self.cost(self.state, control)

        self._clip_state()
        self._check_time()

        return observation, reward, self.done, self.edge

    def _clip_state(self):
        """
        Clip system state > prevent state from exiting the given bounds (self.min, self.max)
        """
        if any(self.state < self.min) or any(self.state > self.max):
            self.state = jnp.array([1, -1]) * self.state
            self.edge = True
        else:
            self.edge = False
        self.state = jnp.clip(self.state, a_min=self.min, a_max=self.max)
    
    def _check_time(self):
        """
        Check if trial end-time has been reached
        """
        if self.t >= self.end_time:
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

    def reset(self, x0=None, T=None, sigma=1, key=jrandom.PRNGKey(randint(0, high=1000))):
        """
        Reset state
        :param x0: new starting state [jnp.array]
        :param T: new trial end-time [int]
        :param sigma: variance of random new starting state [jnp.array]
        :param key: [jrandom.PRNGKey]
        """
        key = random_key(key)
        if x0 == None:
            self.state = jrandom.normal(key, (self.dim,)) * sigma
        else:
            self.state = x0
        
        if self.random_bias:
            self.B = self.B * np.random.choice([-1, 1])

        if T != None:
            self.end_time = T
        
        self.t = 0
        self.done = False
        return self._get_obs(key)
    
    def sample(self, key=None):
        """
        Get random control
        :return: random control value
        """
        key = random_key(key)
        return jrandom.uniform(key, (1,), minval=self.action_space[0,0], maxval=self.action_space[0,1])
    
    def _set_boundary(self, **kwargs):
        self.boundary = kwargs.get('boundary', self.boundary)
        self.min = kwargs.get('min', self.min)
        self.max = kwargs.get('max', self.max)
        self.end_time = kwargs.get('end_time', self.end_time)
    
    def close(self):
        pass
    


def random_key(key, high=1000):
    if key == None:
        return jrandom.PRNGKey(randint(0, high=high))
    else:
        return key



