# -*- coding: utf-8 -*-
"""
Implementation of the Dueling Posterior Sampling algorithm with the Gaussian
process preference credit assignment model. This credit assignment model is
based on Chu and Ghahramani (2005).
"""

import numpy as np
from collections import defaultdict
import itertools
from scipy.optimize import minimize

from Learning_algorithms.DPS_helper_functions import advance, get_state_action_visit_counts


def DPS_GP_preference(time_horizon, hyper_params, env, num_iter, diri_prior = 1, 
            run_str = '', seed = None):
    """
    This function implements the DPS algorithm with a Gaussian process 
    preference credit assignment model over state/action rewards.
    
    Inputs:
        1) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.
        2) hyper_params: the hyperparameters for the GP preference credit 
           assignment model. This is a length-4 list of the form 
           [kernel_variance, kernel_lengthscales, kernel_noise, preference_noise]. 
           While kernel_variance, kernel_noise, and preference_noise are all scalars, 
           kernel_lengthscales can be either a scalar or an array of length 
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
           call to DPS_GP_preference (e.g. hyperparameter values or repetition 
           number), which can be useful for print statements to track progress.
        7) seed: seed for random number generation.
 
    
    Returns: a vector of rewards received as the algorithm runs. This is either
             a) the total rewards from each trajectory rollout, or b) the 
             rewards at every step taken in the environment (the environment
             determines whether a) or b) is used).
    """
    
    # Unpack hyperparameters:
    [kernel_variance, kernel_lengthscales, kernel_noise, preference_noise] = hyper_params
    
    if not seed is None:
        np.random.seed(seed)

    # Numbers of states and actions in the environment:
    num_states = env.nS
    states_per_dim = env.states_per_dim
    num_actions = env.nA
    
    num_sa_pairs = num_states * num_actions   # Number of state/action pairs
    
    # Initialize prior mean for the GP preference model:
    GP_prior_mean = np.zeros(num_sa_pairs)

    # Initialize prior covariance for GP preference model.

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
        kernel_lengthscales = np.concatenate((kernel_lengthscales, [0]))
    
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
    
    GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)

    # Initialize GP preference prior model. Initially, GP model is just the 
    # prior, since we don't have any data.
    
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

    # For each trajectory pair: store difference between how many times the
    # preferred trajectory visits each state/action pair, and how many times 
    # the non-preferred trajectory visits each state/action pair.
    observation_matrix = np.empty((0, num_sa_pairs))

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
        print('GP preference, parameters %s: iteration = %i' % (run_str, \
                                                                iteration + 1))
 
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
                
                next_state, done = env.step(action)
                
                state_sequence[t] = state
                action_sequence[t] = action

                # Tracking rewards for evaluation purposes (in case of 
                # tracking rewards at every single step):
                if not env.store_episode_reward:
                    rewards[reward_count] = env.get_step_reward(state, 
                                action, next_state)
                    reward_count += 1

                # Terminate trajectory if environment turns on "done" flag.
                if done:
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

        visitation_diff = visitations[preference] - visitations[1 - preference]
            
        observation_matrix = np.vstack((observation_matrix, visitation_diff))
   
        # Call feedback function to update the model posterior (with credit
        # assignment via the Gaussian process preference model):
        GP_model = feedback_GP_preference(observation_matrix, GP_prior_cov_inv, 
                                          preference_noise)
    
    # Return performance results:
    return rewards


def feedback_GP_preference(observation_matrix, GP_prior_cov_inv, preference_noise,
                           r_init = []):
    """
    This function updates the GP posterior over rewards based on the new 
    preference data; credit assignment is performed via the Gaussian process 
    preference model.

    Inputs (note: d is the number of state/action pairs; n is the number of 
            data points/observations/rows in the observation matrix):
        1) observation_matrix: n-by-d array, in which each row corresponds 
           to an observation.
        2) GP_prior_cov_inv: the inverse of the prior covariance matrix (d-by-d 
           matrix).
        3) preference_noise: hyperparameter capturing the user's degree of 
           noisiness in specifying preferences.
        4) (Optional) r_init: initial guess for convex optimization; length-d 
           NumPy array when specified.
           
    Output:
        The updated model posterior, represented as a dictionary with keys
        'mean', 'cov_evecs', and 'cov_evals'.
        'mean' is the posterior mean, a length-d NumPy array in which 
        each element corresponds to a state/action pair.
        'cov_evecs' is an d-by-d NumPy array in which each column is an
        eigenvector of the posterior covariance, and 'cov_evals' is a length-d
        array of the eigenvalues of the posterior covariance.

    """
    
    num_sa_pairs = GP_prior_cov_inv.shape[0]   # Number of state/action pairs

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via the Laplace approximation:    
    if r_init == []:
        r_init = np.zeros(num_sa_pairs)    # Initial guess

    res = minimize(preference_GP_objective, r_init, args = (observation_matrix,
                   GP_prior_cov_inv, preference_noise), method = 'L-BFGS-B',
                   jac = preference_GP_gradient)

    # The posterior mean is the solution to the optimization problem:
    post_mean = res.x

    # Approximate inverse of the posterior covariance by evaluating the 
    # objective function's Hessian at the MAP estimate:
    post_cov_inverse = preference_GP_hessian(post_mean, observation_matrix,
                   GP_prior_cov_inv, preference_noise)

    # Calculate the eigenvectors and eigenvalues of the inverse posterior
    # covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov_inverse)

    # Invert the eigenvalues to get the eigenvalues corresponding to the
    # covariance matrix:
    evals = 1 / evals

    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals}
    

def preference_GP_objective(r, observation_matrix, GP_prior_cov_inv, 
                            preference_noise):
    """
    Evaluate the optimization objective function for finding the posterior
    mean of the GP preference model via the Laplace approximation; the 
    posterior mean is the minimum of this (convex) objective function.

    Inputs (note: d is the number of state/action pairs; n is the number of 
            data points/observations/rows in the observation matrix):
        1) r: the "point" at which to evaluate the objective function. This is
           a length-d vector.
        2) observation_matrix: n x d matrix in which every row is an observation.
           Note that all labels are assumed to be 1 by the construction of the
           observation matrix.
        3) GP_prior_cov_inv: the inverse of the prior covariance matrix (d-by-d 
           matrix).
        4) preference_noise: hyperparameter capturing the user's degree of 
           noisiness in specifying preferences.

    Output: the objective function evaluated at the given point (r).
    """

    obj = 0.5 * r @ GP_prior_cov_inv @ r    # Initialize to term from prior

    num_preferences = observation_matrix.shape[0]

    for k in range(num_preferences):   # Go through each preference

        visits = observation_matrix[k]

        z = np.dot(visits, r) / preference_noise

        obj -= np.log(sigmoid(z))

    return obj


def preference_GP_gradient(r, observation_matrix, GP_prior_cov_inv, 
                           preference_noise):
    """
    Evaluate the gradient of the Laplace approximation objective function.

    Inputs: same as in preference_GP_objective.

    Output: the objective function's gradient evaluated at the given point (r).
    """

    grad = GP_prior_cov_inv @ r    # Initialize to term from prior

    num_preferences = observation_matrix.shape[0]

    for k in range(num_preferences):   # Go through each preference
        
        visits = observation_matrix[k]
        z_k = np.dot(visits, r) / preference_noise
        
        grad -= (sigmoid_der(z_k) / (sigmoid(z_k) * preference_noise)) * visits
                 
    return grad


def preference_GP_hessian(r, observation_matrix, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the Hessian of the Laplace approximation objective function.

    Inputs: same as in preference_GP_objective.
    
    Output: the objective function's Hessian matrix evaluated at the given
            point (r).
    """

    # Extract number of preferences in data and number of state/action pairs:
    [num_preferences, num_sa_pairs] = observation_matrix.shape

    Lambda = np.zeros(GP_prior_cov_inv.shape)

    for k in range(num_preferences):   # Go through each preference
        
        visits = observation_matrix[k]

        z_k = np.dot(visits, r) / preference_noise

        c = 1/preference_noise**2 * ( -(sigmoid_2nd_der(z_k)/sigmoid(z_k)) + \
                                     (sigmoid_der(z_k)/sigmoid(z_k))**2)

        Lambda += c * visits.reshape((num_sa_pairs, 1)) @ visits.reshape((1,
                                    num_sa_pairs))

    return GP_prior_cov_inv + Lambda


def sigmoid(x):
    """
    Evaluates the sigmoid function at the specified value.
    Input: x = any scalar
    Output: the sigmoid function evaluated at x.
    """
    # Refine to solve the runtime warning: overflow
    if x>=0:
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))

def sigmoid_der(x):
    """
    Evaluates the sigmoid function's derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's derivative evaluated at x.
    """

    return np.exp(-x) / (1 + np.exp(-x))**2

def sigmoid_2nd_der(x):
    """
    Evaluates the sigmoid function's 2nd derivative at the specified value.
    Input: x = any scalar
    Output: the sigmoid function's 2nd derivative evaluated at x.
    """

    return (-np.exp(-x) + np.exp(-2 * x)) / (1 + np.exp(-x))**3


