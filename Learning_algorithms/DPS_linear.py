# -*- coding: utf-8 -*-
"""
Implementation of the Dueling Posterior Sampling algorithm with Bayesian
linear regression credit assignment.
"""

import numpy as np
from collections import defaultdict

from DPS_helper_functions import advance, get_state_action_visit_counts


def DPS_lin_reg(time_horizon, hyper_params, env, num_iter, diri_prior = 1, 
            run_str = '', seed = None):
    """
    This function implements the DPS algorithm with a Bayesian linear 
    regression credit assignment model over state/action rewards.
    
    Inputs:
        1) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.
        2) hyper_params: the hyperparameters for the linear regression credit 
           assignment model. This is a length-2 list of the form [sigma, 
           lambda], where sigma and lambda are both scalars.
        3) env: the RL environment.
        4) num_iter: the number of iterations of the learning algorithm to run.
           Note that two trajectory rollouts occur per iteration of learning.
        5) diri_prior: parameter for setting the prior of the transition
           dynamics model. For each state/action pair, the Dirichlet prior is
           set to diri_prior * np.ones(num_states), where num_states is the 
           number of states in the MDP.
        6) run_str: if desired, a string with information about the current
           call to DPS_lin_reg (e.g. hyperparameter values or repetition number), 
           which can be useful for print statements to track progress.
        7) seed: seed for random number generation.
 
    
    Returns: a vector of rewards received as the algorithm runs. This is either
             a) the total rewards from each trajectory rollout, or b) the 
             rewards at every step taken in the environment (the environment
             determines whether a) or b) is used).
    """
    
    # Unpack hyperparameters:
    lambd = hyper_params[1]    # Only the lambda hyperparameter is needed for
                               # for initializing the prior
    
    if not seed is None:
        np.random.seed(seed)

    # Numbers of states and actions in the environment:
    num_states = env.nS
    num_actions = env.nA
    
    num_sa_pairs = num_states * num_actions   # Number of state/action pairs

    # Initialize prior mean and covariance for Bayesian linear regression model:
    prior_mean = np.zeros(num_sa_pairs)
    prior_cov = (1 / lambd) * np.eye(num_sa_pairs)
    
    # Eigenvalues and eigenvectors of the prior covariance matrix:
    evals, evecs = np.linalg.eigh(prior_cov) 
    
    # Initially, reward model is just the prior, since we don't have any data.   
    LR_model = {'mean': prior_mean, 'cov_evecs': evecs, 'cov_evals': evals}
    
    # Dirichlet model posterior over state/action transition probabilities.
    # Initially, this is set to the Dirichlet prior, and it's updated after
    # each observed state transition. Note that dirichlet_posterior[state][action] 
    # is a length-num_states array, specifying the probability distribution for
    # transitioning to each possible subsequent state from the given state/action. 
    # Setting diri_prior = 1 gives a uniform prior over transition probabilities.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: \
                                            diri_prior * np.ones(num_states)))

    # For each trajectory pair, store difference between how many times each 
    # state/action pair is visited:
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
        print('Bayesian linear regression, parameters %s: iteration = %i' % \
              (run_str, iteration + 1))
 
        # Sample policies:
        policies = advance(num_policies, dirichlet_posterior, LR_model, 
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

        # Store the difference in state/action visitation counts for the 2 
        # trajectories:
        visitations = []

        for trajectory in trajectories:
            
            visitation_vec = get_state_action_visit_counts(trajectory, 
                                        num_states, num_actions) 
            
            visitations.append(visitation_vec)

        visitation_diff = visitations[1] - visitations[0]
            
        observation_matrix = np.vstack((observation_matrix, visitation_diff))
        
        # Store preference information. Preference gets mapped to 0.5 if 
        # 2nd trajectory is preferred, and -0.5 otherwise.
        preference_labels = np.vstack((preference_labels, 
                        np.reshape([preference - 0.5], (1, 1))))

        # Call feedback function to update the reward model posterior by
        # performing credit assignment via Bayesian linear regression:
        LR_model = feedback_linear(hyper_params, observation_matrix, 
                                   preference_labels)
        
    # Return performance results:
    return rewards


def feedback_linear(LR_prior_params, observation_matrix, preference_labels):
    """
    This function updates the posterior over rewards based on the new 
    preference data, via Bayesian linear regression credit assignment.

    Inputs (note: d is the number of state/action pairs; n is the number of 
            data points/observations/rows in the observation matrix):
        1) LR_prior_params: the hyperparameters for the linear regression credit 
           assignment model. This is a length-2 list of the form [sigma, 
           lambda], where sigma and lambda are both scalars.
        2) observation_matrix: n-by-d array, in which each row corresponds 
           to an observation.
        3) preference_labels: length-n vector, in which each element is the 
           label corresponding to an observation.

    Output:
        The updated model posterior, represented as a dictionary with keys
        'mean', 'cov_evecs', and 'cov_evals'.
        'mean' is the posterior mean, a length-d NumPy array in which 
        each element corresponds to a state/action pair.
        'cov_evecs' is an d-by-d NumPy array in which each column is an
        eigenvector of the posterior covariance, and 'cov_evals' is a length-d
        array of the eigenvalues of the posterior covariance.
    """            

    sigma, lambd = LR_prior_params     # Unpack the prior

    num_sa_pairs = observation_matrix.shape[1]  # Number of state/action pairs

    # Calculate the matrix inverse term used in determining both the posterior
    # mean and covariance:
    intermediate_term = np.transpose(observation_matrix) @ observation_matrix + \
                        sigma**2 * lambd * np.eye(num_sa_pairs)
    intermediate_term = np.linalg.inv(intermediate_term)
            
    # Calculate the posterior mean:
    post_mean = intermediate_term @ np.transpose(observation_matrix) @ \
        preference_labels.flatten()
    
    # Calculate the posterior covariance matrix:
    post_cov = sigma**2 * intermediate_term
    
    # Eigenvectors and eigenvalues of the covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov)

    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals}

