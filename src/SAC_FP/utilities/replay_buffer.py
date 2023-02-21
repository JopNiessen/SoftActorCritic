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

    def __init__(self, buffer_size, obs_size, key, history_size=1, control_size=1, decay=1):
        """
        Initialize replay buffer
        :param buffer_size: buffer size [int]
        :param state_dim: state dimension [int]
        :param action_dim: action dimension [int]
        :param key: PRNGKey
        """
        self.rng = key  # random number generator
        
        self.traj_observation = np.zeros((buffer_size, history_size, obs_size))
        self.traj_control = np.zeros((buffer_size, history_size, control_size))
        self.obs_history = np.zeros((history_size, obs_size))
        self.control_history = np.zeros((history_size+1, control_size))

        self.observation = np.zeros((buffer_size, obs_size))
        self.control = np.zeros((buffer_size, control_size))
        self.reward = np.zeros(buffer_size)
        self.next_observation = np.zeros((buffer_size, obs_size))
        self.done = np.zeros(buffer_size)
        self.importance = np.zeros(buffer_size)
        self.idx = np.arange(0, buffer_size, 1)

        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size
        self.trial_idx = 0
        self.obs_size, self.history_size = obs_size, history_size
        self.control_size = control_size
        self.decay = decay
        self.norm = 1
    
    def feed(self, obs, ctrl, rew, next_obs, done, importance=1):
        self.obs_history[self.trial_idx] = obs
        self.control_history[self.trial_idx+1] = ctrl

        if self.trial_idx >= self.history_size - 1:
            self.traj_observation[self.ptr] = self.obs_history
            self.traj_control[self.ptr] = self.control_history[:-1]
            self.store(obs, ctrl, rew, next_obs, done, importance=importance)
            self.obs_history[:-1] = self.obs_history[1:]
            self.control_history[:-1] = self.control_history[1:]

        else:
            self.trial_idx += 1
        
        if done:
            self.obs_history = np.zeros((self.history_size, self.obs_size))
            self.control_history = np.zeros((self.history_size+1, self.control_size))
            self.trial_idx = 0

    def store(self, obs, ctrl, rew, next_obs, done, importance=1):
        """
        Store datampoint
        :param data_tuple: data of system instance [tuple]
        """
        self.importance = self.importance * self.decay

        self.observation[self.ptr] = obs
        self.control[self.ptr] = ctrl
        self.reward[self.ptr] = rew
        self.next_observation[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.importance[self.ptr] = importance
        self.norm = jnp.sum(self.importance)

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample_batch(self, batch_size):
        """
        Sample from past instances
        :param batch_size: size of sample batch [int]
        """
        batch_size = min(batch_size, self.size)
        self.rng, rng_input = jrandom.split(self.rng)
        idxs = jrandom.choice(rng_input, self.idx, shape=(batch_size,),
                                replace=False, p=self.importance/self.norm)
        # idxs = jrandom.randint(rng_input, shape=(batch_size,),
        #                         minval=0, maxval=self.size)
        return dict(traj_obs = self.traj_observation[idxs],
                    traj_control = self.traj_control[idxs],
                    obs = self.observation[idxs],
                    control = self.control[idxs],
                    reward = self.reward[idxs],
                    next_obs = self.next_observation[idxs],
                    done = self.done[idxs])
        
    def sample_from_buffer(self, batch_size):
        samples = self.sample_batch(batch_size)
        traj_obs = jnp.array(samples['traj_obs'])
        traj_control = jnp.array(samples['traj_control'])
        state = jnp.array(samples['obs'])
        control = jnp.array(samples['control'].reshape(-1, 1))
        reward = jnp.array(samples['reward'].reshape(-1, 1))
        next_state = jnp.array(samples['next_obs'])
        done = jnp.array(samples['done'].reshape(-1, 1))
        return traj_obs, traj_control, state, control, reward, next_state, done
    
    def __len__(self):
        return self.size
