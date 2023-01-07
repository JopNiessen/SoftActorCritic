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
from src.utilities.ReplayBuffer import ReplayBuffer
from src.utilities.Tracker import Tracker

from src.SoftActorCritic.qfunction import QFunction
from src.SoftActorCritic.valuefunction import ValueFunction
from src.SoftActorCritic.policy import PolicyFunction


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
        # set environment
        self.env = env
        keys = jrandom.split(key, 5)
        
        # set default values for the optional arguments
        self.gamma = kwargs.get('gamma', .9)
        self.tau = kwargs.get('tau', 4e-3)
        self.initial_random_steps = kwargs.get('initial_random_steps', 1000)
        self.policy_update_freq = kwargs.get('policy_update_freq', 2)

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
            obs_size = env.observation_space.shape[0]
            ctrl_size = env.action_space.shape[0]
        except:
            if obs_size == None or ctrl_size == None:
                raise Exception('Observation and-or control dim could not be determined')
            else:
                pass

        # build replay buffer
        self.buffer = ReplayBuffer(buffer_size, obs_size, key)
        
        # automatic entropy tuning
        self.target_entropy = -1
        self.log_alpha = jnp.array([0], dtype=float)
        self.alpha_optimizer = optax.adam(lr_alpha)
        self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)

        # actor
        self.actor = PolicyFunction(obs_size, ctrl_size, lr_pi, keys[0], control_limit=self.control_limit)
        
        # v function
        self.VF = ValueFunction(obs_size, lr_v, keys[1])
        self.VF_target = ValueFunction(obs_size, lr_v, keys[1])
        
        # q function
        self.QF1 = QFunction(obs_size + ctrl_size, lr_q, keys[2])
        self.QF2 = QFunction(obs_size + ctrl_size, lr_q, keys[3])

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
            control, _ = self.actor(state, key)
        
        return control
    
    def step(self, state, key, learning=False):
        control = self.get_control(state, key, learning=learning)
        next_state, reward, done, _ = self.env.step(control)
        #reward = jnp.clip(reward/15, a_min=-1)                      # TODO: normalize reward
        
        transition = [state, control, reward, next_state, done]
        if learning:
            self.buffer.store(state, control, reward, next_state, done)
    
        return transition
    
    def q_min(self, state, control):
        q1 = self.QF1(state, control)
        q2 = self.QF2(state, control)
        return jax.lax.min(q1, q2)
    
    def _sample_from_buffer(self, batch_size):
        samples = self.buffer.sample_batch(batch_size)
        state = jnp.array(samples['obs'])
        control = jnp.array(samples['control'].reshape(-1, 1))
        reward = jnp.array(samples['reward'].reshape(-1, 1))
        next_state = jnp.array(samples['next_obs'])
        done = jnp.array(samples['done'].reshape(-1, 1))
        return state, control, reward, next_state, done

    def train_step(self, batch_size, key):
        """Update the model by gradient descent."""
        state, control, reward, next_state, done = self._sample_from_buffer(batch_size)

        keys = jrandom.split(key, len(state))
        new_control, log_prob = jax.vmap(self.actor)(state, keys)
        
        # train alpha (dual problem)
        alpha_loss, alpha_grads = jax.value_and_grad(self.alpha_loss_fn)(self.log_alpha, log_prob)
        updates, self.alpha_opt_state = self.alpha_optimizer.update(alpha_grads, self.alpha_opt_state)
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)
        alpha = jnp.exp(self.log_alpha)
        
        # q function loss
        mask = 1 - done
        v_target = jax.vmap(self.VF_target)(next_state)
        q_target = reward + self.gamma * v_target * mask
        q1_loss, q1_grads = self.QF1.value_and_grad(state, control, q_target)
        q2_loss, q2_grads = self.QF2.value_and_grad(state, control, q_target)
        
        # v function loss
        v_pred = jax.vmap(self.VF)(state)
        q_pred = jax.vmap(self.q_min)(state, new_control)
        v_target = q_pred - alpha * log_prob
        v_loss, v_grads = self.VF.value_and_grad(state, v_target)
        
        if self.step_count % self.policy_update_freq == 0:
            # actor loss
            pi_loss, pi_grads = self.actor.value_and_grad(state, alpha, v_pred, self.q_min, keys)
            
            # train actor
            self.actor.update(pi_grads)
        
            # target update (vf)
            self._update_value_target()
        else:
            pi_loss = 0
            
        # train Q functions
        self.QF1.update(q1_grads)
        self.QF2.update(q2_grads)
        q_loss = q1_loss + q2_loss

        # train V function
        self.VF.update(v_grads)
        return pi_loss, q_loss, v_loss, alpha_loss
    
    def train(self, n_epochs, key, batch_size=100, plotting_interval = 200):
        """Train the agent."""
        
        state = self.env.reset()
        scores = []
        score = 0
        
        for _ in range(n_epochs):
            _, _, reward, next_state, done = self.step(state, key, learning=True)
            #control = self.select_control(state, key)
            #next_state, reward, done = self.step(control)

            # take step
            state = next_state
            score += reward
            self.step_count += 1
            key, subkey = jrandom.split(key)

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            # update actor and critic networks
            if (self.buffer.size >= batch_size and self.step_count > self.initial_random_steps):
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

    # def _update_value_target(self):
    #     """Soft-update: target = tau*local + (1-tau)*target."""
    #     tau = self.tau
    #     base = self.VF.model
    #     target = self.VF_target.model

    #     h1_weight = base.hidden1.weight * tau + target.hidden1.weight * (1-tau)
    #     h1_bias = base.hidden1.bias * tau + target.hidden1.bias * (1-tau)

    #     h2_weight = base.hidden2.weight * tau + target.hidden2.weight * (1-tau)
    #     h2_bias = base.hidden2.bias * tau + target.hidden2.bias * (1-tau)

    #     out_weight = base.out.weight * tau + target.out.weight * (1-tau)
    #     out_bias = base.out.bias * tau + target.out.bias * (1-tau)

    #     target = eqx.tree_at(lambda model: model.hidden1.weight, target, replace=h1_weight)
    #     target = eqx.tree_at(lambda model: model.hidden1.bias, target, replace=h1_bias)
    #     target = eqx.tree_at(lambda model: model.hidden2.weight, target, replace=h2_weight)
    #     target = eqx.tree_at(lambda model: model.hidden2.bias, target, replace=h2_bias)
    #     target = eqx.tree_at(lambda model: model.out.weight, target, replace=out_weight)
    #     target = eqx.tree_at(lambda model: model.out.bias, target, replace=out_bias)

    #     self.VF_target.model = target
    
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