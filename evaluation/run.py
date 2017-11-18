#! /usr/bin/env python3

import os
import argparse
import tb_logger as logger 

from main_algo import train_models, eval_models
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer with Stein Control Variates'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=20)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)

    parser.add_argument('-c', '--coef', type=float, help='Coefficient value',
                        default=1.0)
    parser.add_argument('-u', '--use_lr_adjust', help='whether adaptively adjust lr', type=int, default=0)
    parser.add_argument('-a', '--ada_kl_penalty', help='whether add kl adaptive penalty', type=int, default=1)
    parser.add_argument('-s','--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', help='epochs', type=int, default=20)
    parser.add_argument('-p', '--phi_epochs', help='phi epochs', type=int, default=500)
    parser.add_argument('-m', '--max_timesteps', help='Max timesteps', type=int, default=1000)
    parser.add_argument('-r', '--reg_scale', help='regularization scale on phi function', type=float, default=.0)
    parser.add_argument('-lr', '--phi_lr', help='phi learning_rate', type=float, default=1e-3)#1e-2/np.sqrt(300)
    parser.add_argument('-ph', '--phi_hs', help='phi structure', type=str, default='100x100')
    
    parser.add_argument('-ps', '--policy_size', help='large or small policy size to use', type=str, default='large')
    parser.add_argument('-po', '--phi_obj', help='phi objective function FitQ or MinVar', type=str, default='MinVar')
    parser.add_argument('-sha', '--load_model', 
                        help='if load, save or without doing anything', type=str, default='none')
    args = parser.parse_args()

    args = parser.parse_args()
    if args.load_model == 'save':
        if not os.path.exists('models'):
            os.makedirs('models')

        train_models(**vars(args))
    
    elif args.load_model == 'load':
        if not os.path.exists('eval_data'):
            os.makedirs('eval_data')

        eval_models(**vars(args))
    
    else:
        raise NotImplementedError
    