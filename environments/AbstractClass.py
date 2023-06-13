"""
Abstract class

by J. Niessen
created on: 2023.05.02
"""


import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import gym
from numpy.random import randint

#import environments.numerical_method as nm


class AbstractEnv(gym.Env):
    def __init__(self, dt=0.1, end_time=20, **kwargs):
        """
        Abstract class for dynamical systems
        :param dt: time step size [float]
        :param end_time: end-time of a trial [int]
        """
        self.state = None
        self.done = False

        self.t = 0
        self.dt = dt
        self.end_time = end_time

        #self.num_method = kwargs.get('num_method', nm.euler)
        
        self.dim = kwargs.get('dim', 2)
        self.control_dim = kwargs.get('control_dim', 1)

        self.boundary = kwargs.get('boundary', False)
        self.min = kwargs.get('min', -jnp.ones(self.dim) * jnp.inf)
        self.max = kwargs.get('max', jnp.ones(self.dim) * jnp.inf)

        """System parameters"""
        self.A = kwargs.get('A', jnp.zeros((self.dim, self.dim)))
        self.B = kwargs.get('B', jnp.zeros(self.dim))
        self.C = kwargs.get('C', jnp.identity(self.dim))
        
        noise_v = kwargs.get('noise_v', .1)
        noise_w = kwargs.get('noise_w', .1)
        self.v = kwargs.get('v', jnp.identity(self.dim) * noise_v)        # observation noise
        self.w = kwargs.get('w', jnp.identity(self.dim) * noise_w)        # system noise

        """Cost parameters"""
        self.target = kwargs.get('target', jnp.zeros(self.dim))
        self.F = kwargs.get('F', jnp.identity(self.dim))
        self.G = kwargs.get('G', jnp.identity(self.dim))
        self.R = kwargs.get('R', jnp.identity(self.control_dim))

        """Gym parameters"""
        self.action_space = gym.spaces.Box(low=-4, high=4, shape=(self.control_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim,), dtype=np.float32)
    
    def predict_deriv(self, state, control):
        """
        State derivative
        :param state: state [jnp.array]
        :param control: control [float]
        """
        return jnp.dot(self.A, state) + self.B * control
    
    def _step(self, control, key=None):
        key, subkey = jrandom.split(random_key(key))
        xi = jrandom.normal(key, (self.dim, ))
        
        #self.state = self.num_method(self.predict_deriv, self.state, control, self.dt) + np.sqrt(self.dt) * np.dot(self.w, xi)
        self.state += self.dt * self.predict_deriv(self.state, control) + np.sqrt(self.dt) * np.dot(self.w, xi)
        self.t += self.dt
        
        observation = self._get_obs(subkey)
        reward = -self.cost(self.state, control)

        return observation, reward, self.done, {}

    def _clip_state(self):
        """
        Clip system state > prevent state from exiting the given bounds (self.min, self.max)
        """
        if self.boundary:
            self.state = jnp.clip(self.state, a_min=self.min, a_max=self.max)
    
    def _check_state(self):
        if any(self.state < self.min) or any(self.state > self.max):
            self.done = True
    
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
        return ((x - self.target).T @ self.G @ (x - self.target) + self.R * u**2) * self.dt

    def terminal_cost(self, state):
        """
        Cost in final timestep (t=T)
        :param x: state
        :return: end cost
        """
        x = state
        return x.T @ self.F @ x

    def reset(self, x0=None, T=None, sigma=1, key=None):
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



