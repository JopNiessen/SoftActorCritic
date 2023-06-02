"""
2-Dimensional Linear Quadratic (LQ) system with gaussian noise

by J. Niessen
created on: 2022.10.24
"""

# import libraries
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from numpy.random import randint


# For now, it uses time steps (dt) of 1 sec. With x1 = 'velocity in (m/s)'
class Orbital_SDI:
    def __init__(self, b=1, k=.2, dt=.1, end_time=20):
        """
        This class describes a 2 dimensional linear dynamical system with gaussian noise

        :param b: bias term
        :param k: friction
        :param dt: time step size
        :param time_horizon: end time
        """
        self.state = None
        self.done = False

        self.t = 0
        self.dt = dt
        self.end_time = end_time
        
        self.dim = 2

        """System parameters"""
        self.A = jnp.array([[0, 1], [0, -k]])
        self.B = jnp.array([0, b])
        self.C = jnp.identity(self.dim)
        self.v = jnp.array([[.1, 0], [0, .1]])      # observation noise
        self.w = jnp.array([[0, 0], [0, .2]])       # system noise

        """Cost parameters"""
        self.F = jnp.array([[1, 0], [0, 0]])
        self.G = jnp.array([[1, 0], [0, .001]])
        self.R = .1

        self.reset()
    
    def predict_deriv(self, state, control):
        return jnp.dot(self.A, state) + self.B * control
    
    def state_update(self, control, key):
        xi = jrandom.normal(key, (self.dim, ))
        theta, w = self.state + self.dt * (jnp.dot(self.A, self.state) + self.B * control) + self.dt * np.dot(self.w, xi)
        theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.array([theta, w])

    def step(self, control, key=None):
        key, subkey = jrandom.split(random_key(key))
        self.state = self.state_update(control, key)
        self.t += self.dt
        
        observation = self._get_obs(subkey)
        reward = -self.cost(self.state, control)

        self._check_boundary()

        return observation, reward, self.done, {}

    def _check_boundary(self):
        if self.t >= self.end_time:
            self.done = True

    def _get_obs(self, key):
        """
        Observe the state (x) according to: y(n) = Cx(n) + Vxi
        :return: state observation (y)
        """
        xi = jrandom.normal(key, (self.dim, ))
        obs = np.dot(self.C, self.state) + np.dot(self.v, xi)
        # theta, w = obs
        # x = jnp.cos(theta)
        # y = jnp.sin(theta)
        # return jnp.array([x, y, w])
        return obs

    def cost(self, state, control):
        """
        (Marginal) cost
        :param x: state
        :param u: control
        :return: marginal cost
        """
        x, u = state, control
        #theta, w = state
        #return ((jnp.cos(jnp.pi/4) - jnp.cos(theta))**2 + (jnp.sin(jnp.pi/4) - jnp.sin(theta))**2)*self.dt
        return (x.T @ self.G @ x + self.R * u**2)*self.dt

    def terminal_cost(self, state):
        """
        Cost in final timestep (t=T)
        :param x: state
        :return: end cost
        """
        x = state
        return x.T @ self.F @ x

    def reset(self, x0=None, T=None, key=None):
        """
        Reset state
        :param x0: initial state
        """
        key = random_key(key)
        if x0 == None:
            theta = jrandom.uniform(key, minval=-1, maxval=1) * jnp.pi
            w = jrandom.normal(key)
            self.state = jnp.array([theta, w])
        else:
            self.state = x0
        
        if T != None:
            self.end_time = T

        self.t = 0
        self.done = False
        return self._get_obs(key)
    
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



