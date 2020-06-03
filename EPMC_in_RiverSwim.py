# -*- coding: utf-8 -*-
"""
Run the Every-Visit Preference Monte Carlo (EPMC) algorithm in the RiverSwim 
environment.

This algorithm, which we use as a baseline, is described in "A Policy Iteration 
Algorithm for Learning from Preference-Based Feedback" by C. Wirth and J. 
Furnkranz (2013) and in "Efficient Preference-based Reinforcement Learning" 
by C. Wirth (2017).
"""

import numpy as np
import scipy.io as io
import os

from Envs.RiverSwim import RiverSwimPreferenceEnv
from Learning_algorithms.EPMC import EPMC


# Define constants:
time_horizon = 50
num_iter = 100     # Number of preference queries/trajectory pairs per run

# User noise models take the form [noise_level, noise_type]; see the function
# in the RiverSwim class that generates preferences for details of how these
# are used.
user_noise_params = [[1000, 1], [1, 1], [0.5, 1], [0.1, 1], [0.5/50, 2]]

run_nums = 100    # Number of times to run the algorithm

# Folder for saving results:
output_folder = 'EPMC/'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# Hyperparameters ranges to test with EPMC:
etas = np.round(np.arange(0.1, 1, 0.1), decimals = 2)
alphas = np.round(np.arange(0.1, 1, 0.1), decimals = 2)

# Run EPMC for each user noise model.    
for user_noise in user_noise_params:

    # Instantiate the environment:
    env = RiverSwimPreferenceEnv(user_noise)       
    
    # Run EPMC with each pair of hyperparameter values.
    for eta, alpha, run_num in [(eta, alpha, run_num) \
                             for eta in etas for alpha in alphas \
                             for run_num in range(run_nums)]:    

        # String to use in status updates:
        run_str = 'eta %2.1f, alpha %2.1f, user noise %s, run %i' %  \
                    (eta, alpha, user_noise, run_num)
        
        # Run algorithm:
        hyper_params = [alpha, eta]
        rewards = EPMC(time_horizon, hyper_params, env, num_iter, 
                          run_str = run_str)
        
        # Save results from this algorithm run:
        
        if user_noise[1] < 2:    # Logistic noise cases
            noise_str = str(user_noise[0])
        else:                    # Linear noise case
            noise_str = 'linear'
        
        output_filename = output_folder + 'Iter_' + str(num_iter) + '_alpha_'  \
            + str(alpha) + '_eta_' + str(eta) + '_noise_' + noise_str + \
            '_run_' + str(run_num) + '.mat'
        
        io.savemat(output_filename, {'rewards': rewards, 'num_iter': num_iter, 
                  'hyper_params': hyper_params, 'user_noise_model': user_noise})
        
