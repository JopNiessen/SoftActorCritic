"""
Replay Buffer
"""

# import global libraries
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom


class ReplayBuffer():
    """
    Replay buffer for off-policy learing

    Alteration on publicly available module. Original module can be found on: https://github.com/chisarie/jax-agents/blob/master/jax_agents/common/data_processor.py
    """

    def __init__(self, buffer_size, obs_size, key, control_size=1):
        """
        Initialize replay buffer
        :param buffer_size: buffer size [int]
        :param state_dim: state dimension [int]
        :param action_dim: action dimension [int]
        :param key: PRNGKey
        """
        self.rng = key  # rundom number generator

        self.observation = np.zeros((buffer_size, obs_size))
        self.control = np.zeros((buffer_size, control_size))
        self.reward = np.zeros(buffer_size)
        self.next_observation = np.zeros((buffer_size, obs_size))
        self.done = np.zeros(buffer_size)

        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size

    def store(self, obs, ctrl, rew, next_obs, done):
        """
        Store datampoint
        :param data_tuple: data of system instance [tuple]
        """
        self.observation[self.ptr] = obs
        self.control[self.ptr] = ctrl
        self.reward[self.ptr] = rew
        self.next_observation[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample_batch(self, batch_size):
        """
        Sample from past instances
        :param batch_size: size of sample batch [int]
        """
        batch_size = min(batch_size, self.size)
        self.rng, rng_input = jrandom.split(self.rng)
        idxs = jrandom.randint(rng_input, shape=(batch_size,),
                                minval=0, maxval=self.size)
        return dict(obs = self.observation[idxs],
                    control = self.control[idxs],
                    reward = self.reward[idxs],
                    next_obs = self.next_observation[idxs],
                    done = self.done[idxs])
    
    def __len__(self):
        return self.size




