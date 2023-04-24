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
from src.SoftActorCriticRNN.utilities.ReplayBuffer import ReplayBuffer
from src.utilities.Tracker import Tracker

from src.SoftActorCriticRNN.qfunction import RQFunction
from src.SoftActorCriticRNN.valuefunction import RValueFunction
from src.SoftActorCriticRNN.policy import PolicyFunctionRNN

import src.SoftActorCriticRNN.valuefunction as vf
import src.SoftActorCriticRNN.qfunction as qf
import src.SoftActorCriticRNN.policy as pi


class SACAgent:
    """
    Recurrent Soft Actor-Critic (SAC) agent
    """
    def __init__(
            self,
            env,
            buffer_size: int,
            key: jrandom.PRNGKey,
            **kwargs
        ):
        """
        Required parameters
        :param env: environment [gym style]
        :param buffer_size: buffer size [int]
        :param key: key [jrandom.PRNGKey]

        Optional parameters
        :param hidden_size: number of nodes in recurrent layers (default: 8) [int]
        :param history_size: numer of past instances saved in replay buffer (default: 1) [int]

        :param gamma: discount factor (default: .9) [float]
        :param tau: rate of exponential moving target value average (default: 4e-3) [float]
        :param memory_decay: discount factor for replay memory (default: 0.) [float]

        :param initial_random_steps: number of initial random controls (default: 1000) [int]
        :param policy_update_freq: policy update frequency (default: 2) [int]
        :param epochs_per_step: number of epochs per environment step (default: 1) [int]
        
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

        # recurrence parameters
        self.hidden_size = kwargs.get('hidden_size', 8)
        self.history_size = kwargs.get('history_size', 1)
        
        # set default values for the optional arguments
        self.gamma = kwargs.get('gamma', .9)
        self.tau = kwargs.get('tau', 4e-3)
        self.initial_random_steps = kwargs.get('initial_random_steps', 1000)
        self.policy_update_freq = kwargs.get('policy_update_freq', 2)
        self.epochs_per_step = kwargs.get('epochs_per_step', 1)

        # set learning rates
        lr = kwargs.get('lr', 2e-3)
        lr_pi = kwargs.get('lr_pi', lr)
        lr_q = kwargs.get('lr_q', lr)
        lr_v = kwargs.get('lr_v', lr)
        lr_alpha = kwargs.get('lr_alpha', lr)

        # environment parameters
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
        
        # RNN type
        self.rnntype = kwargs.get('RNN_type', 'WilsonCowan')

        """Replay buffer"""
        memory_decay = kwargs.get('memory_decay', 0)
        self.buffer = ReplayBuffer(buffer_size, self.obs_size, key, decay=1-memory_decay, history_size=self.history_size)
        
        """automatic entropy tuning"""
        self.target_entropy = -1
        self.log_alpha = jnp.array([0], dtype=float)
        self.alpha_optimizer = optax.adam(lr_alpha)
        self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)

        # store losses
        self.tracker = Tracker(['pi_loss', 'q_loss', 'v_loss', 'alpha_loss'])
        
        # set number of training iterations
        self.step_count = 0

        """Actor model"""
        self.actor_model = pi.generate_instance(self.obs_size+self.ctrl_size, self.ctrl_size, lr_pi, keys[0], **kwargs)

        """Critic models"""
        self.value_model = vf.generate_instance(self.obs_size + self.ctrl_size, lr_v, keys[1], **kwargs)
        self.value_target_fn = vf.generate_instance(self.obs_size + self.ctrl_size, lr_v, keys[1], **kwargs)[0]

        self.q1_model = qf.generate_instance(self.obs_size + self.ctrl_size, self.ctrl_size, lr_q, keys[2], **kwargs)
        self.q2_model = qf.generate_instance(self.obs_size + self.ctrl_size, self.ctrl_size, lr_q, keys[3], **kwargs)
    
    def alpha_loss_fn(self, log_alpha, log_prob):
        """Entropy loss function"""
        return -jnp.mean(jnp.exp(log_alpha) * (log_prob + self.target_entropy))
    
    def get_control(self, state, control, hidden_state, key, learning=False):
        """Select control based on current state and policy"""
        if self.step_count < self.initial_random_steps and learning:
            control = jrandom.uniform(key, shape=(1,), minval=-1, maxval=1) * self.control_limit
        else:
            control, _, hidden_state = self.actor_model[0].forward_step(state, control, hidden_state, key)
        
        return control, hidden_state
    
    def step(self, state, control, hidden_state, key, learning=False):
        """Perform a single environment step"""
        control, hidden_state = self.get_control(state, control, hidden_state, key, learning=learning)
        next_state, reward, done, _ = self.env.step(control, key=key)
        
        if learning:
            self.buffer.feed(state, control, reward, next_state, done)
    
        return state, control, hidden_state, reward, next_state, done
        
    def _sample_from_buffer(self, batch_size):
        """Sample a batch from the replay buffer"""
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
        """Perform a single training step"""
        traj_state, traj_control, state, control, reward, next_state, done = self._sample_from_buffer(batch_size)
        
        actor = self.actor_model[0]
        qf1 = self.q1_model[0]
        qf2 = self.q2_model[0]
        
        keys = jrandom.split(key, len(state))
        new_control, log_prob = jax.vmap(actor)(traj_state, traj_control, keys)
        
        """update alpha (dual problem)"""
        alpha_loss, alpha_grads = jax.value_and_grad(self.alpha_loss_fn)(self.log_alpha, log_prob)
        updates, self.alpha_opt_state = self.alpha_optimizer.update(alpha_grads, self.alpha_opt_state)
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)
        alpha = jnp.exp(self.log_alpha)
        
        """Q-function gradients"""
        mask = 1 - done
        next_control_seq = jnp.hstack((traj_control, control.reshape((batch_size, 1, 1))))[:,1:,:]
        next_state_seq = jnp.hstack((traj_state, next_state.reshape((batch_size, 1, 2))))[:,1:,:]
        v_target = jax.vmap(self.value_target_fn)(next_state_seq, next_control_seq)
        q_target = reward + self.gamma * v_target * mask
        q1_loss, params_q1 = qf.make_step(self.q1_model, traj_state, traj_control, control, q_target)
        q2_loss, params_q2 = qf.make_step(self.q2_model, traj_state, traj_control, control, q_target)

        """Value function gradients"""
        value_fn, _, _ = self.value_model
        v_pred = jax.vmap(value_fn)(traj_state, traj_control)
        q1_pred = jax.vmap(qf1)(traj_state, traj_control, new_control)
        q2_pred = jax.vmap(qf2)(traj_state, traj_control, new_control)
        q_pred = jax.vmap(jax.lax.min)(q1_pred, q2_pred)
        v_target = q_pred - alpha * log_prob
        v_loss, params_v = vf.make_step(self.value_model, traj_state, traj_control, v_target)
        
        """update policy"""
        if self.step_count % self.policy_update_freq == 0:
            # update actor
            pi_loss, params_pi = pi.make_step(self.actor_model, traj_state, traj_control, alpha, v_pred, qf1, qf2, keys)
            self.actor_model = params_pi
        
            # update value target
            self.value_target_fn = self._update_value_target(value_fn, self.value_target_fn)
        else:
            pi_loss = 0
        
        #print(f'iteration={self.step_count}')

        """update q-functions"""
        self.q1_model = params_q1
        self.q2_model = params_q2
        q_loss = q1_loss + q2_loss
        
        """update value function"""
        self.value_model = params_v

        return pi_loss, q_loss, v_loss, alpha_loss
    
    def train(self, n_epochs, key, batch_size=100, plotting_interval = 200, record=False):
        """
        Train the agent for n_epochs

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train for
        key : jax.random.PRNGKey
            Random key for sampling
        batch_size : int, optional
            Batch size for training, by default 100
        plotting_interval : int, optional
            Interval for plotting, by default 200
        record : bool, optional
            Whether to record the training, by default False
        """
        state = self.env.reset(key=key)
        self.buffer.clear()
        scores = []
        score = 0

        hidden_state = jnp.zeros(self.hidden_size)
        control = 0

        for it in range(n_epochs):
            _, control, hidden_state, reward, next_state, done = self.step(state, control, hidden_state, key, learning=True)

            # take step
            state = next_state
            score += reward
            self.step_count += 1
            key, subkey = jrandom.split(key)

            if done:
                state = self.env.reset(key=key)
                control = 0
                hidden_state = jnp.zeros(self.hidden_size)
                scores.append(score)
                score = 0

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
                            
        self.env.close()
    
    def _update_value_target(self, base, target):
        tau = self.tau

        for idx, (base_layer, target_layer) in enumerate(zip(base.general_layers, target.general_layers)):
            weight = base_layer.weight * tau + target_layer.weight * (1 - tau)
            bias = base_layer.bias * tau + target_layer.bias * (1-tau)

            target = eqx.tree_at(lambda model: model.general_layers[idx].weight, target, replace=weight)
            target = eqx.tree_at(lambda model: model.general_layers[idx].bias, target, replace=bias)

        if self.rnntype == 'WilsonCowan':
            A_weight = base.EncoderCell.A.weight * tau + target.EncoderCell.A.weight * (1 - tau)
            B_weight = base.EncoderCell.B.weight * tau + target.EncoderCell.B.weight * (1 - tau)
            target = eqx.tree_at(lambda model: model.EncoderCell.A.weight, target, replace=A_weight)
            target = eqx.tree_at(lambda model: model.EncoderCell.B.weight, target, replace=B_weight)
        elif self.rnntype == 'GRU':
            weight_ih = base.EncoderCell.cell.weight_ih * tau + target.EncoderCell.cell.weight_ih * (1 - tau)
            weight_hh = base.EncoderCell.cell.weight_hh * tau + target.EncoderCell.cell.weight_hh * (1 - tau)
            target = eqx.tree_at(lambda model: model.EncoderCell.cell.weight_ih, target, replace=weight_ih)
            target = eqx.tree_at(lambda model: model.EncoderCell.cell.weight_hh, target, replace=weight_hh)
        
        return target
    
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
