#! /usr/bin/env python3

import os
import argparse
import tb_logger as logger 

from main_algo import main
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer with Stein Control Variates'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_iterations', type=int, help='Number of iterations to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of batch_size per training batch',
                        default=10000)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                        '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument('-c', '--coef', type=float, help='Stein control variate coefficient value',
                        default=1.0)
    parser.add_argument('-u', '--use_lr_adjust', help='whether adaptively adjust lr', type=int, default=0)
    parser.add_argument('-a', '--ada_kl_penalty', help='whether add kl adaptive penalty', type=int, default=1)
    parser.add_argument('-s','--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', help='epochs', type=int, default=20)
    parser.add_argument('-p', '--phi_epochs', help='phi epochs', type=int, default=500)
    parser.add_argument('-mt', '--max_timesteps', help='Max timesteps', type=int, default=1000)
    parser.add_argument('-r', '--reg_scale', help='regularization scale on phi function', type=float, default=.0)
    parser.add_argument('-lr', '--phi_lr', help='phi learning_rate', type=float, default=0.0005)
    parser.add_argument('-ph', '--phi_hs', 
                        help='phi structure, default 100x100 for mlp', 
                        type=str, default='100x100')
    
    parser.add_argument('-ps', '--policy_size', 
			help='large or small policy size to use, \
            use small for Ant, Humanoid and HumanoidStandup', 
			type=str, default='large')
    parser.add_argument('-po', '--phi_obj', help='phi objective \
            function FitQ or MinVar', type=str, default='MinVar')
    
    args = parser.parse_args()

    # logs
    dir_name = os.path.join('dartml_data', 'env=%s/'%(args.env_name))
    
    if args.coef == 0.:
        dir_name += 'PPO-%s'%(datetime.now().strftime('%m_%d_%H:%M:%S'))
    else:
        dir_name += 'Stein-PPO_Phi_obj=%s-%s'%(args.phi_obj, \
            datetime.now().strftime('%m_%d_%H:%M:%S'))

    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.environ["DARTML_LOGDIR"]=dir_name
    logger.set_logdir(dir_name)

    args = parser.parse_args()
    main(**vars(args))
