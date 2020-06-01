# -*- coding: utf-8 -*-
"""
Run DPS algorithm in the RiverSwim environment, using Bayesian linear
regression credit assignment.
"""

import scipy.io as io
import os

from Envs.RiverSwim import RiverSwimPreferenceEnv
from Learning_algorithms.DPS_linear import DPS_lin_reg


# Define constants:
time_horizon = 50
num_iter = 200     # Number of preference queries/trajectory pairs per run

# User noise models take the form [noise_level, noise_type]; see the function
# in the RiverSwim class that generates preferences for details of how these
# are used.
user_noise_params = [[1000, 1], [1, 1], [0.5, 1], [0.1, 1], [0.5/50, 2]]

run_nums = 100    # Number of times to run the algorithm

# Folder for saving results:
output_folder = 'DPS_linear/'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# Hyperparameters to use for Bayesian linear regression:
sigma = 0.5
lambd = 0.1
hyper_params = [sigma, lambd]

# Run DPS with each user noise model.
for user_noise in user_noise_params:

    # Instantiate the environment:
    env = RiverSwimPreferenceEnv(user_noise)   

    for run_num in range(run_nums):     
        
        # String to use in status updates:
        run_str = 'sigma = %1.2f, lambda = %1.2f, user noise = %s, run = %i' %   \
                  (sigma, lambd, user_noise, run_num)
        
        # Run algorithm:
        rewards = DPS_lin_reg(time_horizon, hyper_params, env, num_iter, 
                          run_str = run_str)
        
        # Save results from this algorithm run:
        
        if user_noise[1] < 2:    # Logistic noise cases
            noise_str = str(user_noise[0])
        else:                    # Linear noise case
            noise_str = 'linear'
        
        output_filename = output_folder + 'Iter_' + str(num_iter) + '_params_' + \
                        str(sigma) + '_' + str(lambd) + '_noise_' + noise_str + \
                        '_run_' + str(run_num) + '.mat'
        
        io.savemat(output_filename, {'rewards': rewards, 'num_iter': num_iter, 
                  'hyper_params': hyper_params, 'user_noise_model': user_noise})
        
