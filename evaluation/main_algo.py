#! /usr/bin/env python3
import os
import gym
import pickle
import random

import numpy as np
import tb_logger as logger 

import scipy.signal
from gym import wrappers
from utils import Scaler
from policy import Policy
from datetime import datetime
from value_function import NNValueFunction

def init_gym(env_name):
   
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def run_episode(env, policy, scaler, max_timesteps, animate=False):
   
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


def run_policy(env, policy, scaler, num_episodes, max_timesteps):
  
    total_steps = 0
    trajectories = []
    traj_len_list = []

    for itr in range(num_episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, \
                    policy, scaler, 
                    max_timesteps=max_timesteps)
        
        total_steps += observes.shape[0]
        
        traj_len_list.append(len(observes))

        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled) 
    logger.record_dicts({
        "_MeanReward":np.mean([t['rewards'].sum() for t in trajectories]),
        'Steps': total_steps,})

    return trajectories, traj_len_list


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
   
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
   
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
   
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
    
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, episode):
    
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



def train_models(env_name, num_episodes, 
        gamma, lam, kl_targ, 
        coef, use_lr_adjust, 
        ada_kl_penalty, seed, 
        epochs, phi_epochs,
        max_timesteps, reg_scale,
        phi_lr, phi_hs,
        policy_size, 
        phi_obj, load_model): 

    env, obs_dim, act_dim = init_gym(env_name)
    set_global_seeds(seed) 
    env.seed(seed)
    env._max_episode_steps = max_timesteps
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    aigym_path = os.path.join('log-files/', env_name, now)
    env = wrappers.Monitor(env, aigym_path, force=True, video_callable=False)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim)
    policy = Policy(obs_dim, act_dim, 
            kl_targ,epochs, 
            phi_epochs, 
            policy_size=policy_size,
            phi_hidden_sizes=phi_hs,
            reg_scale=reg_scale,
            lr_phi=phi_lr,
            phi_obj=phi_obj)

    
    run_policy(env, policy, 
            scaler, num_episodes, 
            max_timesteps=max_timesteps)
    
    episode = 0
    for _ in range(200):
        trajectories, traj_len_list = run_policy(env, policy, scaler, 
                            num_episodes, max_timesteps=max_timesteps)
    
        num_traj = len(trajectories)
    
        episode += len(trajectories)
        add_value(trajectories, val_func)  
        add_disc_sum_rew(trajectories, gamma)  
        add_gae(trajectories, gamma, lam) 
    
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        
        policy.update(load_model, observes, actions, advantages,
                use_lr_adjust, ada_kl_penalty, c=0.)  # update policy
        val_func.fit(observes, disc_sum_rew) 

    # Save models
    policy.save_policy()
    val_func.save_val_func()
    logger.log("saved model")


def eval_models(env_name, num_episodes, 
        gamma, lam, kl_targ, 
        coef, use_lr_adjust, 
        ada_kl_penalty, seed, 
        epochs, phi_epochs,
        max_timesteps, reg_scale,
        phi_lr, phi_hs,
        policy_size, 
        phi_obj, load_model): 

    env, obs_dim, act_dim = init_gym(env_name)
    set_global_seeds(seed) 
    env.seed(seed)
    env._max_episode_steps = max_timesteps
    obs_dim += 1  
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S") 
    aigym_path = os.path.join('log-files/', env_name, now)
    env = wrappers.Monitor(env, aigym_path, force=True, video_callable=False)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim)
    policy = Policy(obs_dim, act_dim, 
            kl_targ,epochs, 
            phi_epochs, 
            policy_size=policy_size,
            phi_hidden_sizes=phi_hs,
            reg_scale=reg_scale,
            lr_phi=phi_lr,
            phi_obj=phi_obj)

 
    logger.log("loading model")
    load_dir = "models/"
    policy.load_model(load_dir)
    val_func.load_val_model(load_dir)


    run_policy(env, policy, 
            scaler, num_episodes, 
            max_timesteps=max_timesteps)
    
    episode = 0

    trajectories, traj_len_list = run_policy(env, policy, scaler, 
                              num_episodes, max_timesteps=max_timesteps)
    
    num_traj = len(trajectories)
    logger.log("Avg Length %d total Length %d"%( \
            np.mean(traj_len_list), \
            np.sum(traj_len_list)))
    
    episode += len(trajectories)
    add_value(trajectories, val_func)  
    add_disc_sum_rew(trajectories, gamma)  
    add_gae(trajectories, gamma, lam) 
    
    #Split data into validation and training data
    random.shuffle(trajectories)
    t_trajectories = trajectories[:int(len(trajectories)/2)]
    v_trajectories = trajectories[int(len(trajectories)/2):]

    t_observes, t_actions, t_advantages, t_disc_sum_rew = build_train_set(t_trajectories)
    v_observes, v_actions, v_advantages, v_disc_sum_rew = build_train_set(v_trajectories)

    sub_folder = "eval_data/%s_%s_data_seed=%d_max-steps=%d"%(\
                        env_name, phi_obj, 
                        seed, max_timesteps)
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)
    
    # save original gradient
    mc_grad_info = policy.get_batch_gradient(v_observes, v_actions, v_advantages, c=0.)
    mc_grad_info['traj_lens'] = traj_len_list
    with open(sub_folder+'/mc_num_episode=%d.pkl'%(num_episodes), 'wb') as fp:
        pickle.dump(mc_grad_info, fp)
    
    
    policy.update(load_model, t_observes, t_actions, t_advantages,
            use_lr_adjust, ada_kl_penalty, c=1)  # update policy
            
    stein_grad_info = policy.get_batch_gradient(v_observes, \
                    v_actions, v_advantages, c=1.)

    
    stein_grad_info['traj_lens'] = traj_len_list
    with open(sub_folder+'/stein_num_episode=%d.pkl'%(num_episodes), 'wb') as fp:
        pickle.dump(stein_grad_info, fp)
