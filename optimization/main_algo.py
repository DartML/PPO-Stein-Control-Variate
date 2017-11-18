#! /usr/bin/env python3
"""
Stein PPO: Sample-efficient Policy Optimization with Stein Control Variate

Motivated by the Stein’s identity, Stein PPO extends the previous 
control variate methods used in REINFORCE and advantage actor-critic 
by introducing more general action-dependent baseline functions.
Details see the following papers:

Stein PPO:
https://arxiv.org/pdf/1710.11198.pdf

Distributed PPO:
https://arxiv.org/abs/1707.02286

Proximal Policy Optimization Algorithms
https://arxiv.org/pdf/1707.06347.pdf

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

Code modified from this Github repo: https://github.com/pat-coady/trpo

This GitHub repo is also helpful.
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import os
import gym
import random

import numpy as np
import tb_logger as logger 

import scipy.signal
from gym import wrappers
from utils import Scaler
from policy import Policy
from datetime import datetime
from value_function import NNValueFunction


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, max_timesteps, animate=False): 
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    for _ in range(max_timesteps):
        
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
        if done:
            break

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, batch_size, max_timesteps):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        episodes: total episodes to run
        max_timesteps: max timesteps per episode to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []

    while total_steps < batch_size:
        observes, actions, rewards, unscaled_obs = run_episode(env, \
                    policy, scaler, max_timesteps=max_timesteps)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)


    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    
    logger.record_dicts({
        "_MeanReward":np.mean([t['rewards'].sum() for t in trajectories]),
        'Steps': total_steps,})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew):
    """ Log batch statistics """

    logger.record_dicts({
        '_mean_obs': np.mean(observes),
        '_min_obs': np.min(observes),
        '_max_obs': np.max(observes),
        '_mean_act': np.mean(actions),
        '_max_act': np.max(actions),
        '_std_act': np.mean(np.var(actions, axis=0)),
        '_mean_adv': np.mean(advantages),
        '_min_adv': np.min(advantages),
        '_max_adv': np.max(advantages),
        '_std_adv': np.var(advantages),
        '_mean_discrew': np.mean(disc_sum_rew),
        '_min_discrew': np.min(disc_sum_rew),
        '_max_discrew': np.max(disc_sum_rew),
        '_std_discrew': np.var(disc_sum_rew)})
    
    logger.dump_tabular()


def main(env_name, num_iterations, gamma, lam, kl_targ, 
        batch_size,hid1_mult, policy_logvar, coef, use_lr_adjust, ada_kl_penalty, 
        seed, epochs, phi_epochs, max_timesteps,
        reg_scale, phi_lr, 
        phi_hs, 
        policy_size, 
        phi_obj):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_iterations: maximum number of iterations to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
        coef: coefficient of Stein control variate
        use_lr_adjust: whether adjust lr based on kl
        ada_kl_penalty: whether adjust kl penalty
        max_timesteps: maximum time steps per trajectory
        reg_scale: regularization coefficient 
        policy_size: policy network size
        phi_obj: FitQ or MinVar
    """

    env, obs_dim, act_dim = init_gym(env_name)
    set_global_seeds(seed)
    env.seed(seed)
    env._max_episode_steps = max_timesteps
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
    aigym_path = os.path.join('log-files/', env_name, now)
    env = wrappers.Monitor(env, aigym_path, force=True, video_callable=False)
    
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    
    policy = Policy(obs_dim, act_dim, kl_targ, 
            hid1_mult, policy_logvar,
            epochs, phi_epochs, 
            policy_size=policy_size,
            phi_hidden_sizes=phi_hs,
            c_ph=coef, 
            reg_scale=reg_scale,
            lr_phi=phi_lr,
            phi_obj=phi_obj)
    
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, batch_size=1000, max_timesteps=max_timesteps)

    for _ in range(num_iterations):
        logger.log("\n#Training Iter %d"%(_))
        logger.log("Draw Samples..")
        
        trajectories = run_policy(env, policy, scaler, 
            batch_size=batch_size, max_timesteps=max_timesteps) 
        
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew)

        logger.log("Starting Training...")
        policy.update(observes, actions, advantages, \
                use_lr_adjust, ada_kl_penalty)  # update policy
 
        val_func.fit(observes, disc_sum_rew)  # update value function

        logger.log('--------------------------------\n')

    policy.close_sess()
    val_func.close_sess()