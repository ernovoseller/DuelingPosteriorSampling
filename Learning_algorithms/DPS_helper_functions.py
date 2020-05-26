# -*- coding: utf-8 -*-
"""
This script implements some helper functions for the Dueling Posterior Sampling
algorithm.
"""

import numpy as np

import sys
if "../" not in sys.path:
  sys.path.append("../")
from ValueIteration import value_iteration


def get_state_action_visit_counts(trajectory, num_states, num_actions):
    """
    This function returns a vector recording how many times each state-action 
    pair is visited in a trajectory.
    
    Inputs: (Note: h is the trajectory horizon)
        1) trajectory: this is a list of the form [states, actions], where
           states is a length-h (or length h + 1) NumPy array listing all the 
           states encountered in the trajectory, and actions is a length-h 
           NumPy array listing all of the actions encountered in the trajectory.
        2) num_states: number of states in the MDP.
        3) num_actions: number of actions in the MDP.
    
    Returns: a length num_states * num_actions NumPy array recording how many 
             times each state-action pair is visited in the trajectory. Note:
             state i (i from 0 to num_states - 1) and action j (j from 0 to 
             num_actions - 1) corresponds to state/action pair 
             num_actions * i + j.
    """
    
    # Extract state and action sequences:
    states = trajectory[0].astype(int)
    actions = trajectory[1].astype(int)

    # Initialize state/action visit counts:
    visit_counts = np.zeros(num_states * num_actions)
    
    # Go through each step in the trajectory. We assume that len(actions)
    # is equal to the time_horizon.
    for t in range(len(actions)):

        # Increment visit count for the state/action pair at this time step.
        visit_counts[num_actions * states[t] + actions[t]] += 1

    return visit_counts.reshape((1,num_states * num_actions))


def advance(num_policies, dirichlet_posterior, reward_posterior, num_states, 
                        num_actions, time_horizon):
    """
    Draw a specified number of samples from the model posteriors over the 
    environment (i.e., the transition dynamics and rewards). For each sampled 
    environment, run value iteration to obtain the optimal policy given the
    sampled dynamics and rewards.
    
    This function assumes that the reward model posterior is a Gaussian 
    distribution, specified by its mean and covariance.

    Inputs: (note: d = num_states * num_actions, the number of state/action pairs)
        1) num_policies: the number of samples to draw from the posterior; a
           positive integer.
        2) dirichlet_posterior: the model posterior over transition dynamics
           parameters: dirichlet_posterior[state][action] is a length-num_states
           array of the Dirichlet parameters for the given state and action.
           These give the probability distribution of transitioning to each 
           possible subsequent state from the given state and action.
        3) reward_posterior: this is the reward model posterior, represented 
           by a dictionary of the form {'mean': post_mean, 'cov_evecs': evecs,
           'cov_evals': evals}; post_mean is the posterior mean, a length-d
           NumPy array. cov_evecs is an d-by-d NumPy array in
           which each column is an eigenvector of the posterior covariance,
           and evals is a length-d array of the eigenvalues of the posterior
           covariance.
        4) num_states: number of states in the MDP.
        5) num_actions: number of actions in the MDP.
        6) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.

    Output:
        1) policies: this is a length-num_policies list, in which each element
           is a policy. A policy is represented by a NumPy array of size 
           time_horizon x num_states x num_actions, in which policy[t][s][a] 
           is the probability that the policy takes action a in state s at 
           time-step t.

    """    
    
    policies = []
    
    for i in range(num_policies):
        
        # Sample state transition dynamics from Dirichlet posterior:
        dynamics_sample = []
        
        for state in range(num_states):
            
            dynamics_sample_ = []
            
            for action in range(num_actions):
        
                dynamics_sample_.append(np.random.dirichlet(dirichlet_posterior[state][action]))
                
            dynamics_sample.append(dynamics_sample_)
    
        # Sample rewards from the reward model posterior:
        mean = reward_posterior['mean']
        
        X = np.random.normal(size = num_states * num_actions)
        R = mean + reward_posterior['cov_evecs'] @   \
                np.diag(np.sqrt(reward_posterior['cov_evals'])) @ X

        R = R.reshape([num_states, num_actions])
              
        # Value iteration to determine policy:             
        policies.append(value_iteration(dynamics_sample, R, num_states, 
                                        num_actions, epsilon = 0, 
                                        H = time_horizon)[0])
        
    return policies

