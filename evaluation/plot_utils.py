import numpy as np 
import pickle
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# max_length = 1000
def cal_traj_gradient(traj_len_list, sample_grads):

    total_length = 0
    traj_grad = []

    for traj_len in  traj_len_list:
        cur_traj_grad = sample_grads[total_length:(total_length+traj_len)]
        total_length += traj_len
        traj_grad.append(np.mean(cur_traj_grad, axis=0))
    
    print(len(traj_len_list))
    # assert total_length == max_length

    return traj_grad
        

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
            
            # traj_grad = cal_traj_gradient(traj_data['traj_lens'],sample_grads)
            
            mc_grads.append(sample_grads)
            mc_phi_loss.append(traj_data['phi_loss'])

    stein_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'stein_num_episode=%d.pkl'%i)
        with open(file_path, 'rb') as f:
            traj_data = pickle.load(f)
            sample_grads = np.concatenate([traj_data['mu_grad'], 
                                        traj_data['sigma_grad']], axis=1)
            
            # traj_grad = cal_traj_gradient(traj_data['traj_lens'], sample_grads)

            stein_grads.append(sample_grads)
            stein_phi_loss.append(traj_data['phi_loss'])

    return mc_grads, stein_grads, mc_phi_loss, stein_phi_loss


def gen_index(indices, max_length):
    total_indices = []

    for index in indices:
        total_indices.append(np.arange(index*max_length, (index+1)* max_length))
    
    return np.concatenate(total_indices, axis=0)