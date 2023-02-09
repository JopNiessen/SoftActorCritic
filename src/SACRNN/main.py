"""
Soft Actor-Critic agent
"""

# import libraries
import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import optax

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt

# import local libraries
from src.SACRNN.utilities.ReplayBuffer import ReplayBuffer
from src.utilities.Tracker import Tracker

from src.SACRNN.qfunction import QFunction
from src.SACRNN.valuefunction import ValueFunction
from src.SACRNN.policy import PolicyFunction, RecurrentPolicyFunction


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent
    """
    def __init__(self, env, buffer_size: int, key: jrandom.PRNGKey, **kwargs):
        """
        Initialization of the SAC agent
        :param env: environment [gym styled]
        :param buffer_size: buffer size [int]
        :param key: key [jrandom.PRNGKey]
        
        :param gamma: discount factor (default: .9) [float]
        :param tau: rate of exponential moving target value average (default: 4e-3) [float]
        :param initial_random_steps: number of initial random controls (default: 1000) [int]
        :param policy_update_freq: policy update frequency (default: 2) [int]
        
        :param lr: learning rate (default: 2e-3) [float]
        :param lr_pi: policy learning rate (default: 2e-3) [float]
        :param lr_q: Q learning rate (default: 2e-3) [float]
        :param lr_v: value learning rate (default: 2e-3) [float]
        :param lr_alpha: entropy learning rate (default: 2e-3) [float]

        :param control_limit: maximal control magnitude (default: 1) [float]
        :param obs_size: number of observables [int]
        :param ctrl_size: number of control variables [int]
        """
        self.recording = {'state':[], 'angle':[], 'force':[], 'time':[]}   #TODO: temporary

        # set environment
        self.env = env
        keys = jrandom.split(key, 5)
        
        # set default values for the optional arguments
        self.gamma = kwargs.get('gamma', .9)
        self.tau = kwargs.get('tau', 4e-3)
        self.initial_random_steps = kwargs.get('initial_random_steps', 1000)
        self.policy_update_freq = kwargs.get('policy_update_freq', 2)
        self.epochs_per_step = kwargs.get('epochs_per_step', 1)

        # recurrence
        self.history_size = kwargs.get('history_size', 8)

        # set learning rates
        lr = kwargs.get('lr', 2e-3)
        lr_pi = kwargs.get('lr_pi', lr)
        lr_q = kwargs.get('lr_q', lr)
        lr_v = kwargs.get('lr_v', lr)
        lr_alpha = kwargs.get('lr_alpha', lr)

        # set remaining parameters
        self.control_limit = kwargs.get('control_limit', 1.)
        self.obs_size = kwargs.get('obs_size', None)
        self.ctrl_size = kwargs.get('ctrl_size', None)

        # set dimensions
        try:
            self.obs_size = env.observation_space.shape[0]
            self.ctrl_size = env.action_space.shape[0]
        except:
            if self.obs_size == None or self.ctrl_size == None:
                raise Exception('Observation and-or control dim could not be determined')
            else:
                pass

        # build replay buffer
        memory_decay = kwargs.get('memory_decay', 0)
        self.buffer = ReplayBuffer(buffer_size, self.obs_size, key, history_size=self.history_size, decay=1-memory_decay)
        
        # automatic entropy tuning
        self.target_entropy = -1
        self.log_alpha = jnp.array([0], dtype=float)
        self.alpha_optimizer = optax.adam(lr_alpha)
        self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)

        # actor
        self.actor = PolicyFunction(self.obs_size, self.ctrl_size, lr_pi, keys[0], control_limit=self.control_limit)
        self.rec_actor = RecurrentPolicyFunction(self.obs_size + self.ctrl_size, self.ctrl_size, lr_pi, key=keys[0])
        self.rec_actor_hidden = jnp.zeros(self.obs_size)
        
        # v function
        self.VF = ValueFunction(self.obs_size, lr_v, keys[1])
        self.VF_target = ValueFunction(self.obs_size, lr_v, keys[1])
        
        # q function
        self.QF1 = QFunction(self.obs_size + self.ctrl_size, lr_q, keys[2])
        self.QF2 = QFunction(self.obs_size + self.ctrl_size, lr_q, keys[3])

        # store losses
        self.tracker = Tracker(['pi_loss', 'q_loss', 'v_loss', 'alpha_loss'])
        
        # set number of training iterations
        self.step_count = 0
    
    def alpha_loss_fn(self, log_alpha, log_prob):
        return -jnp.mean(jnp.exp(log_alpha) * (log_prob + self.target_entropy))
    
    def get_control(self, state, key, learning=False):
        if self.step_count < self.initial_random_steps and learning:
            control = jrandom.uniform(key, shape=(1,), minval=-1, maxval=1) * self.control_limit
        else:
            #control, _ = self.actor(state, key)
            control, _, self.rec_actor_hidden = self.rec_actor.step(state, self.prev_control, self.rec_actor_hidden, key)
        
        return control
    
    def step(self, state, key, learning=False):
        control = self.get_control(state, key, learning=learning)
        next_state, reward, done, _ = self.env.step(control)
        #reward = jnp.clip(reward/15, a_min=-1)                      # TODO: normalize reward
        
        if learning:
            self.buffer.feed(state, control, reward, next_state, done)
    
        return state, control, reward, next_state, done
    
    def q_min(self, state, control):
        q1 = self.QF1(state, control)
        q2 = self.QF2(state, control)
        return jax.lax.min(q1, q2)
    
    def _sample_from_buffer(self, batch_size):
        samples = self.buffer.sample_batch(batch_size)
        traj_obs = jnp.array(samples['traj_obs'])
        traj_control = jnp.array(samples['traj_control'])
        state = jnp.array(samples['obs'])
        control = jnp.array(samples['control'].reshape(-1, 1))
        reward = jnp.array(samples['reward'].reshape(-1, 1))
        next_state = jnp.array(samples['next_obs'])
        done = jnp.array(samples['done'].reshape(-1, 1))
        return traj_obs, traj_control, state, control, reward, next_state, done

    def train_step(self, batch_size, key):
        traj_obs, traj_control, state, control, reward, next_state, done = self._sample_from_buffer(batch_size)

        keys = jrandom.split(key, len(state))
        #new_control, log_prob = jax.vmap(self.actor)(state, keys)
        new_control, log_prob = jax.vmap(self.rec_actor)(traj_obs, traj_control, keys)
        
        # update alpha (dual problem)
        alpha_loss, alpha_grads = jax.value_and_grad(self.alpha_loss_fn)(self.log_alpha, log_prob)
        updates, self.alpha_opt_state = self.alpha_optimizer.update(alpha_grads, self.alpha_opt_state)
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)
        alpha = jnp.exp(self.log_alpha)
        
        # Q-function gradients
        mask = 1 - done
        v_target = jax.vmap(self.VF_target)(next_state)
        q_target = reward + self.gamma * v_target * mask
        q1_loss, q1_grads = self.QF1.value_and_grad(state, control, q_target)
        q2_loss, q2_grads = self.QF2.value_and_grad(state, control, q_target)
        
        # value function gradients
        v_pred = jax.vmap(self.VF)(state)
        q_pred = jax.vmap(self.q_min)(state, new_control)
        v_target = q_pred - alpha * log_prob
        v_loss, v_grads = self.VF.value_and_grad(state, v_target)
        
        if self.step_count % self.policy_update_freq == 0:
            # update actor
            pi_loss, pi_grads = self.actor.value_and_grad(state, alpha, v_pred, self.q_min, keys)
            self.actor.update(pi_grads)

            rpi_loss, rpi_grads = self.rec_actor.value_and_grad(traj_obs, traj_control, alpha, v_pred, self.q_min, keys)
            self.rec_actor.update(rpi_grads)
        
            # update value target
            self._update_value_target()
        else:
            pi_loss = 0
            rpi_loss = 0
            
        # update Q-functions
        self.QF1.update(q1_grads)
        self.QF2.update(q2_grads)
        q_loss = q1_loss + q2_loss

        # update value function
        self.VF.update(v_grads)
        return rpi_loss, q_loss, v_loss, pi_loss#alpha_loss
    
    def train(self, n_epochs, key, batch_size=100, plotting_interval = 200, record=False):

        state = self.env.reset()
        scores = []
        score = 0
        
        for _ in range(n_epochs):
            _, control, reward, next_state, done = self.step(state, key, learning=True)

            # take step
            state = next_state
            score += reward
            self.step_count += 1
            key, subkey = jrandom.split(key)

            if done:
                state = self.env.reset(sigma=jnp.array([3,1]))
                scores.append(score)
                score = 0
                self.prev_control = control
                self.rec_actor_hidden = jnp.zeros(self.obs_size)
            else:
                self.prev_control = jnp.array([0])

            # update actor and critic networks
            if (self.buffer.size >= batch_size and self.step_count > self.initial_random_steps):
                for _ in range(self.epochs_per_step):
                    pi_loss, q_loss, v_loss, alpha_loss = self.train_step(batch_size, subkey)
                    self.tracker.add([pi_loss, q_loss, v_loss, alpha_loss])
            
            # plot
            if self.step_count % plotting_interval == 0:
                self._plot(
                    self.step_count,
                    scores, 
                    self.tracker('pi_loss'),
                    self.tracker('q_loss'),
                    self.tracker('v_loss'),
                    self.tracker('alpha_loss')
                )
            
            if record:
                self.recording['state'].append(state)
                params = self.actor.model.mu_layer.weight
                self.recording['angle'].append(jnp.arctan2(params[0,0], params[0,1]))
                self.recording['force'].append(jnp.linalg.norm(params[0]))
                self.recording['time'].append(self.env.t)
                
        self.env.close()
    
    def _update_value_target(self):
        tau = self.tau
        base = self.VF.model
        target = self.VF_target.model

        for idx, (base_layer, target_layer) in enumerate(zip(base.general_layers, target.general_layers)):
            weight = base_layer.weight * tau + target_layer.weight * (1 - tau)
            bias = base_layer.bias * tau + target_layer.bias * (1-tau)

            target = eqx.tree_at(lambda model: model.general_layers[idx].weight, target, replace=weight)
            target = eqx.tree_at(lambda model: model.general_layers[idx].bias, target, replace=bias)
        self.VF_target.model = target
    
    def _plot(self, epoch, scores, pi_loss, q_loss, v_loss, alpha_loss):
        """Plot the training progresses."""
        def subplot(loc, title, values):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (151, f'epoch {epoch}. score: {np.mean(scores[-10:])}', scores),
            (152, 'Actor loss', pi_loss),
            (153, 'Q loss', q_loss),
            (154, 'Value loss', v_loss),
            (155, 'Entropy loss', alpha_loss),
        ]
        
        clear_output(True)
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()

