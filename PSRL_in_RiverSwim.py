# -*- coding: utf-8 -*-
"""
Run the posterior sampling RL algorithm (PSRL) in the RiverSwim environment.

The PSRL algorithm is described in "(More) Efficient Reinforcement Learning via 
Posterior Sampling," by I. Osband, B. Van Roy, and D. Russo (2013). It learns 
from numerical rewards, rather than preferences.
"""

import scipy.io as io
import os

from Envs.RiverSwim import RiverSwimEnv
from Learning_algorithms.PSRL_numerical_rewards import PSRL


# Define constants:
time_horizon = 50
num_iter = 400     # Number of iterations of the learning algorithm. This is 
                   # twice the number of iterations of the preference-based
                   # algorithms, since PSRL rolls out one trajectory/episode
                   # per learning iteration, while the preference-based 
                   # algorithms roll out two; thus, the number of episodes is
                   # kept consistent.

run_nums = 100    # Number of times to run the algorithm

# Folder for saving results:
output_folder = 'PSRL/'

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# Hyperparameters to use for the PSRL algorithm:
NG_params = [1, 1, 1, 1]      # Prior parameters for normal-gamma reward model

# Instantiate the environment:
env = RiverSwimEnv()   

# Run PSRL algorithm:
for run_num in range(run_nums):     
    
    # String to use in status updates:
    run_str = '%s, run %i' % (NG_params, run_num)
    
    # Run algorithm:
    rewards = PSRL(time_horizon, NG_params, env, num_iter, run_str = run_str)
                      
    # Save results from this algorithm run:
    output_filename = output_folder + 'Iter_' + str(num_iter) + '_params_' + \
                        str(NG_params[0]) + '_' + str(NG_params[1]) \
                         + '_' + str(NG_params[2]) + '_' + str(NG_params[3]) + \
                        '_run_' + str(run_num) + '.mat'
    
    io.savemat(output_filename, {'rewards': rewards, 'num_iter': num_iter, 
              'NG_params': NG_params})
    
