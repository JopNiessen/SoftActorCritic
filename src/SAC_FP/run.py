"""
SAC main training loop
"""

import jax.random as jrandom

# import local libraries
from src.SAC_FP.utilities.preprocessing import *
from src.SAC_FP.train import *


def run_SAC(params, env, buffer, n_steps, batch_size, key, online=True, **kwargs):
    gamma = kwargs.get('gamma', .9)
    policy_update_freq = kwargs.get('policy_update_freq', 2)
    initial_random_steps = kwargs.get('initial_random_steps', 200)
    epochs_per_step = kwargs.get('epochs_per_step', 1)
    control_scale = kwargs.get('control_scale', 1)
    state_scale = kwargs.get('state_scale', 5)
    reward_scale = kwargs.get('reward_scale', state_scale * env.dt)

    key, subkey = jrandom.split(key)

    obs = env._get_obs(key)
    control = 0
    pi_hid = obs
    for step in range(n_steps):
        # make step in environment
        if online:
            pi_fn, _, _ = params[0]

            if buffer.size < initial_random_steps:
                control = jrandom.uniform(key, (1,), minval=-1, maxval=1)   #jrandom.normal(key, (1,))/4
            else:
                control, pi_hid = pi_fn.predict_step(normalize(obs, state_scale), control, pi_hid)
            new_obs, rew, done, _ = env.step(control * control_scale)

            buffer.feed(normalize(obs, state_scale), control, normalize(rew, reward_scale), normalize(new_obs, state_scale), done)

            if done:
                obs = env.reset()
                control = 0
                pi_hid = obs
            else:
                obs = new_obs
        
        # train step
        if buffer.size > batch_size:
            for _ in range(epochs_per_step):
                params, loss = train_step_rec(params, buffer, batch_size, step, subkey, gamma=gamma, policy_update_freq=policy_update_freq)
                pi_loss, q_loss, v_loss, alpha_loss = loss
                print(f'step={step:5.0f}\tpi={pi_loss:.5f}\tq={q_loss:.5f}\tv={v_loss:.5f}\talpha={alpha_loss:.5f}')

                # update key
                subkey, _ = jrandom.split(subkey)
        
        key, subkey = jrandom.split(key)

    return params, env, buffer