import numpy as np 
import pickle
import os 
import errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_sample_grads(batch_range, prefix_dir):
    file_dir = prefix_dir

    # load mc traj
    stein_phi_loss = []
    mc_phi_loss = []
    mc_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'mc_num_episode=%d.pkl'%i)
        with open(file_path, 'rb') as f:
            traj_data = pickle.load(f)
            sample_grads = np.concatenate([traj_data['mu_grad'], 
                                        traj_data['sigma_grad']], axis=1)
            
            
            mc_grads.append(sample_grads)
            mc_phi_loss.append(traj_data['phi_loss'])

    stein_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'stein_num_episode=%d.pkl'%i)
        with open(file_path, 'rb') as f:
            traj_data = pickle.load(f)
            sample_grads = np.concatenate([traj_data['mu_grad'], 
                                        traj_data['sigma_grad']], axis=1)

            stein_grads.append(sample_grads)
            stein_phi_loss.append(traj_data['phi_loss'])

    return mc_grads, stein_grads, mc_phi_loss, stein_phi_loss


def gen_index(indices, max_length):
    total_indices = []

    for index in indices:
        total_indices.append(np.arange(index*max_length, (index+1)* max_length))
    
    return np.concatenate(total_indices, axis=0)


if __name__ == '__main__':

    batch_range= range(10, 70, 10)

    env_name = 'Walker2d-v1'

    try:
        seeds = [int(x) for x in input("Enter seeds of evaluation saved data, seperated by space\n").split()]
    except (SyntaxError, ValueError):
        seeds = list(range(13, 253, 30))
    if not seeds:
        seeds = list(range(13, 253, 30))        
    print(seeds)

    try:
        phi_obj = input("Enter type of evaluation, FitQ or MinVar: \n")
    except (SyntaxError, ValueError):
        print("no choice about evaluation type")
        phi_obj = 'FitQ'

    try: 
        max_timesteps = input("Enter max_timesteps: \n")
    except (SyntaxError, ValueError):
        print("no choice about max_timesteps")        
        max_timesteps = 50
    
    k = 20000
    plot_stein_loss = []
    plot_mc_loss = []
                                 
    for seed in seeds:

        prefix_dir = 'max_timesteps=%s_eval_data/%s_%s_data_seed=%d_max-steps=%s'%(max_timesteps, env_name, phi_obj, seed, max_timesteps)
    
        print(prefix_dir) 

        # This is gradient for each trajectory
        mc_x = []
        stein_x = []
        plot_stein_vars = []
        plot_mc_vars = []
    
        mc_grads, stein_grads, mc_phi_loss, \
                    stein_phi_loss = load_sample_grads(batch_range, prefix_dir)

        for mc_grad, stein_grad in zip(mc_grads, stein_grads):
        
            mc_x.append(len(mc_grad))
            stein_x.append(len(stein_grad))
            print(len(mc_grad))
            # Calculate MSE/Variance
            mc_vars = []
            mc_num_traj = len(mc_grad)
            for kk in range(k):
                
                indices = np.random.choice(mc_num_traj, int(mc_num_traj/2), replace=False)
                total_indices = np.arange(0,  mc_num_traj)
                mask = np.zeros(total_indices.shape, dtype=bool)
                mask[indices] = True
            
                mc_grad = np.array(mc_grad)
                mc_var = (np.mean(mc_grad[total_indices[mask]], axis=0) - \
                        np.mean(mc_grad[total_indices[~mask]], axis=0)) ** 2
                mc_vars.append(np.sum(mc_var))

            plot_mc_vars.append(np.mean(mc_vars))

    
            stein_vars = []
            stein_num_traj = len(stein_grad)

            for kk in range(k):
                
                indices = np.random.choice(stein_num_traj, int(stein_num_traj/2), replace=False)
                total_indices = np.arange(0, stein_num_traj)
                mask = np.zeros(total_indices.shape, dtype=bool)
                mask[indices] = True
                stein_grad = np.array(stein_grad)
                stein_var = (np.mean(stein_grad[total_indices[mask]], axis=0) - \
                             np.mean(stein_grad[total_indices[~mask]], axis=0)) ** 2
                stein_vars.append(np.sum(stein_var))
        
            plot_stein_vars.append(np.mean(stein_vars))

        print (seed)
        print (mc_x)
        print (stein_x)
        print (np.log(plot_stein_vars))
        print (np.log(plot_mc_vars))
        plt.plot(np.log(mc_x), np.log(plot_mc_vars), label='mc')
        plt.plot(np.log(stein_x), np.log(plot_stein_vars), label='state-baseline')
        plt.legend()
        mkdir_p('results')
        plt.savefig('results/' + '%s_avg_variance_seed=%s_max-steps=%s_phi_obj=%s.pdf'%(env_name, seed, max_timesteps, phi_obj))
        plt.gcf().clear()




    


        
