# -*- coding: utf-8 -*-
"""
Implementation of the Dueling Posterior Sampling algorithm with Gaussian
process regression credit assignment.
"""

import numpy as np
from collections import defaultdict
import itertools

from DPS_helper_functions import advance, get_state_action_visit_counts


def DPS_GPR(time_horizon, hyper_params, env, num_iter, diri_prior = 1, 
            run_str = '', seed = None):
    """
    This function implements the DPS algorithm with a Gaussian process 
    regression credit assignment model over state/action rewards.
    
    Inputs:
        1) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.
        2) hyper_params: the hyperparameters for the GP regression credit 
           assignment model. This is a length-3 list of the form 
           [kernel_variance, kernel_lengthscales, kernel_noise]. While 
           kernel_variance and kernel_noise are both scalars, kernel_lengthscales
           can be either a scalar or an array of length 
           num_states * num_actions + 1. If it is a scalar, then the kernel 
           lengthscale is set to this number in every state space dimension,
           and the lengthscale is set to 0 over actions. If it is an array, 
           then the last element corresponds to the lengthscale over actions, 
           while the previous elements correspond to the lengthscale in each 
           state space dimension.
        3) env: the RL environment.
        4) num_iter: the number of iterations of the learning algorithm to run.
           Note that two trajectory rollouts occur per iteration of learning.
        5) diri_prior: parameter for setting the prior of the transition
           dynamics model. For each state/action pair, the Dirichlet prior is
           set to diri_prior * np.ones(num_states), where num_states is the 
           number of states in the MDP.
        6) run_str: if desired, a string with information about the current
           call to DPS_GPR (e.g. hyperparameter values or repetition number), 
           which can be useful for print statements to track progress.
        7) seed: seed for random number generation.
 
    
    Returns: a vector of rewards received as the algorithm runs. This is either
             a) the total rewards from each trajectory rollout, or b) the 
             rewards at every step taken in the environment (the environment
             determines whether a) or b) is used).
    """
    
    # Unpack hyperparameters:
    [kernel_variance, kernel_lengthscales, kernel_noise] = hyper_params
    
    if not seed is None:
        np.random.seed(seed)

    # Numbers of states and actions in the environment:
    num_states = env.nS
    states_per_dim = env.states_per_dim
    num_actions = env.nA
    
    num_sa_pairs = num_states * num_actions   # Number of state/action pairs
    
    # Initialize prior mean for the GP model:
    GP_prior_mean = np.zeros(num_sa_pairs)

    # Initialize prior covariance for GP model.

    # Map for converting state-action pair index to indices within each state
    # and action dimension:
    ranges = []
    for i in range(len(states_per_dim)):
        ranges.append(np.arange(states_per_dim[i]))

    ranges.append(np.arange(num_actions))

    state_action_map = list(itertools.product(*ranges))
    
    # If kernel_lengthscales is a scalar, we convert it to a vector as described
    # in input 4) in the large comment at the top of this function.
    if np.isscalar(kernel_lengthscales):
        kernel_lengthscales = kernel_lengthscales * np.ones(len(states_per_dim))
        kernel_lengthscales = np.concatenate(kernel_lengthscales, [0])
    
    GP_prior_cov = kernel_variance * np.ones((num_sa_pairs, num_sa_pairs))

    for i in range(num_sa_pairs):

        x1 = state_action_map[i]

        for j in range(num_sa_pairs):

            x2 = state_action_map[j]

            # Consider lengthscales in each state space dimension.
            for dim, lengthscale in enumerate(kernel_lengthscales):

                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((x2[dim] - x1[dim]) / lengthscale)**2)

                elif lengthscale == 0 and x1[dim] != x2[dim]:

                    GP_prior_cov[i, j] = 0

    GP_prior_cov += kernel_noise * np.eye(num_sa_pairs)

    # Gaussian process prior:
    GP_prior = {'mean': GP_prior_mean, 'cov': GP_prior_cov}
    
    # Initially, GP model is just the prior, since we don't have any data:
    
    # Eigenvalues and eigenvectors of the prior covariance matrix:
    evals, evecs = np.linalg.eigh(GP_prior_cov)    
    GP_model = {'mean': GP_prior_mean, 'cov_evecs': evecs, 'cov_evals': evals}
    
    # Dirichlet model posterior over state/action transition probabilities.
    # Initially, this is set to the Dirichlet prior, and it's updated after
    # each observed state transition. Note that dirichlet_posterior[state][action] 
    # is a length-num_states array, specifying the probability distribution for
    # transitioning to each possible subsequent state from the given state/action. 
    # Setting diri_prior = 1 gives a uniform prior over transition probabilities.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: \
                                            diri_prior * np.ones(num_states)))

    # Store how many times each trajectory visits each state/action pair:
    observation_matrix = np.empty((0, num_sa_pairs))
    
    # Preference labels corresponding to the observations:
    preference_labels = np.empty((0, 1))

    num_policies = 2     # Number of policies to sample per learning iteration
    
    # To store results (for evaluation purposes only):
    if env.store_episode_reward:   # Store total reward for each trajectory
        rewards = np.empty(num_iter * num_policies)
    else:    # Store reward at each step within each trajectory
        rewards = np.empty(num_iter * time_horizon * num_policies)
    
    reward_count = 0   # Counts how many values in the "rewards" variable 
                       # defined above have been populated

    """
    Here is where the learning algorithm begins.
    """
    for iteration in range(num_iter):
        
        # Print status:
        print('GPR, parameters %s: iteration = %i' % (run_str, iteration + 1))
 
        # Sample policies:
        policies = advance(num_policies, dirichlet_posterior, GP_model, 
                        num_states, num_actions, time_horizon)
    
        # Roll out trajectories using these policies:
        trajectories = []
        
        for policy in policies:    # Roll out 2 action sequences
    
            state = env.reset()
                    
            state_sequence = np.empty(time_horizon + 1)
            action_sequence = np.empty(time_horizon)
            
            for t in range(time_horizon):  
                
                action = np.random.choice(num_actions, p = policy[t, state, :])
                
                next_state, _, _ = env.step(action)
                
                state_sequence[t] = state
                action_sequence[t] = action

                # Tracking rewards for evaluation purposes (in case of 
                # tracking rewards at every single step):
                if not env.store_episode_reward:
                    rewards[reward_count] = env.get_step_reward(state, 
                                action, next_state)
                    reward_count += 1

                # Terminate trajectory if environment turns on "done" flag.
                if env.done:
                    state_sequence = state_sequence[: t + 2]
                    action_sequence = action_sequence[: t + 1]
                    
                    break
                
                # Update state transition posterior:
                dirichlet_posterior[state][action][next_state] += 1

                state = next_state                    
            
            state_sequence[-1] = next_state
            trajectories.append([state_sequence, action_sequence]) 

        # Tracking rewards for evaluation purposes (in case of tracking
        # rewards just over entire episodes):
        if env.store_episode_reward:

            rewards[reward_count] = env.get_episode_return()
            reward_count += 1
            
        # Obtain a preference between the 2 trajectories:
        preference = env.get_trajectory_preference(trajectories[0], 
                    trajectories[1])

        # This only matters if using deterministic preferences (in this case, 
        # there can be a tie between two trajectories. In this case, we skip 
        # updating the reward posterior):
        if preference == 0.5:
            continue

        # Store state/action visitation counts for the 2 trajectories:
        for trajectory in trajectories:
            
            visitation_vec = get_state_action_visit_counts(trajectory, 
                                        num_states, num_actions) 
            
            observation_matrix = np.vstack((observation_matrix, visitation_vec))
        
        # Store preference information:
        preference_labels = np.vstack((preference_labels, 
                        np.reshape([0.5 - preference, preference - 0.5], (2, 1))))
   
        # Call feedback function to update the reward model posterior by
        # performing credit assignment via Gaussian process regression:
        GP_model = feedback_GPR(GP_prior, observation_matrix, preference_labels)
    
    # Return performance results:
    return rewards


def feedback_GPR(GP_prior, observation_matrix, preference_labels, 
                 obs_noise = 1e-5):
    """
    This function updates the GP posterior over rewards based on the new 
    preference data, via Gaussian process regression credit assignment.

    Inputs (note: d is the number of state/action pairs; n is the number of 
            data points/observations/rows in the observation matrix):
        1) GP_prior: the prior, represented as a dictionary with keys 'mean'
           and 'cov'. These are the prior mean (length-d array) and prior 
           covariance (d-by-d matrix) respectively.
        2) observation_matrix: n-by-d array, in which each row corresponds 
           to an observation.
        3) preference_labels: length-n vector, in which each element is the 
           label corresponding to an observation.
        4) obs_noise: observation noise hyperparameter (denoted 
           $\sigma_{\epsilon}$ in Appendix B.3.2 of the paper). This can be set 
           to a very small number (e.g., 1e-5 by default) to ensure that the 
           posterior covariance matrix is positive definite.

    Output:
        The updated model posterior, represented as a dictionary with keys
        'mean', 'cov_evecs', and 'cov_evals'.
        'mean' is the posterior mean, a length-d NumPy array in which 
        each element corresponds to a state/action pair.
        'cov_evecs' is an d-by-d NumPy array in which each column is an
        eigenvector of the posterior covariance, and 'cov_evals' is a length-d
        array of the eigenvalues of the posterior covariance.

    """
    
    # Unpack the prior:        
    prior_mean = GP_prior['mean']
    prior_cov = GP_prior['cov']

    num_samples = len(preference_labels)   # Number of data points so far
            
    # Calculate the posterior mean:
    K_rR = prior_cov @ np.transpose(observation_matrix)
    K_R = observation_matrix @ K_rR
    intermediate_term = K_rR @ np.linalg.inv(K_R +  \
                                             obs_noise * np.eye(num_samples))
    
    post_mean = prior_mean + intermediate_term @ \
        (preference_labels.flatten() - observation_matrix @ prior_mean)
    
    # Calculate the posterior covariance matrix:
    post_cov = prior_cov - intermediate_term @ np.transpose(K_rR)

    # Eigenvectors and eigenvalues of the covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov)

    # Return the GP model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals}
    
