# -*- coding: utf-8 -*-
"""
Implementation of the Dueling Posterior Sampling algorithm with Bayesian
logistic regression credit assignment.
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

from DPS_helper_functions import advance, get_state_action_visit_counts


def DPS_log_reg(time_horizon, hyper_params, env, num_iter, diri_prior = 1, 
            run_str = '', seed = None):
    """
    This function implements the DPS algorithm with a Bayesian logistic
    regression credit assignment model over state/action rewards.
    
    Inputs:
        1) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.
        2) hyper_params: the hyperparameters for the logistic regression credit 
           assignment model. This is a length-2 list of the form 
           [prior_covariance, cov_scale], where the prior covariance and 
           covariance scale are both scalars.
        3) env: the RL environment.
        4) num_iter: the number of iterations of the learning algorithm to run.
           Note that two trajectory rollouts occur per iteration of learning.
        5) diri_prior: parameter for setting the prior of the transition
           dynamics model. For each state/action pair, the Dirichlet prior is
           set to diri_prior * np.ones(num_states), where num_states is the 
           number of states in the MDP.
        6) run_str: if desired, a string with information about the current
           call to DPS_log_reg (e.g. hyperparameter values or repetition number), 
           which can be useful for print statements to track progress.
        7) seed: seed for random number generation.
 
    
    Returns: a vector of rewards received as the algorithm runs. This is either
             a) the total rewards from each trajectory rollout, or b) the 
             rewards at every step taken in the environment (the environment
             determines whether a) or b) is used).
    """
    
    # Unpack hyperparameters:
    [prior_covariance, cov_scale] = hyper_params
    
    if not seed is None:
        np.random.seed(seed)

    # Numbers of states and actions in the environment:
    num_states = env.nS
    num_actions = env.nA
    
    num_sa_pairs = num_states * num_actions   # Number of state/action pairs

    # Initialize prior mean and covariance for Bayesian logistic regression model:
    prior_mean = np.zeros(num_sa_pairs)
    prior_cov = prior_covariance * np.eye(num_sa_pairs)
    
    # Prior model, without covariance scaling:
    LR_prior_model = {'mean': prior_mean, 'cov': prior_cov}
    
    # Eigenvalues and eigenvectors of the prior covariance matrix:
    evals, evecs = np.linalg.eigh(prior_cov) 
    
    # Initially, reward model is just the prior, since we don't have any data.   
    LR_model = {'mean': prior_mean, 'cov_evecs': evecs, \
                'cov_evals': cov_scale * evals}
    
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
        print('Bayesian logistic regression, parameters %s: iteration = %i' % \
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
        # performing credit assignment via Bayesian logistic regression:
        LR_model = feedback_logistic(LR_prior_model, observation_matrix, 
                            preference_labels, cov_scale)
        
    # Return performance results:
    return rewards


def feedback_logistic(LR_prior_model, observation_matrix, preference_labels,
                      cov_scale, r_init = []):
    """
    This function updates the posterior over rewards based on the new 
    preference data, via Bayesian logistic regression credit assignment.
    The Bayesian logistic regression posterior is calculated via the Laplace
    approximation.

    Inputs (note: d is the number of state/action pairs; n is the number of 
            data points/observations/rows in the observation matrix):
        1) LR_prior_model: the prior (without covariance scaling via cov_scale), 
           represented as a dictionary with keys 'mean' and 'cov'. These are
           the prior mean (length-d array) and prior covariance (d-by-d matrix)
           respectively.
        2) observation_matrix: n-by-d array, in which each row corresponds 
           to an observation.
        3) preference_labels: length-n vector, in which each element is the 
           label corresponding to an observation.
        4) cov_scale: covariance scaling hyperparameter; a positive scalar.
        5) (Optional) r_init: initial guess for convex optimization; length-d 
           NumPy array when specified.
           
    Output:
        The updated model posterior, represented as a dictionary with keys
        'mean', 'cov_evecs', and 'cov_evals'. 'mean' is the posterior mean, a
        length-d NumPy array in which each element corresponds to a state/action
        pair. 'cov_evecs' is an d-by-d NumPy array in which each column is an
        eigenvector of the posterior covariance, and 'cov_evals' is a length-d
        array of the eigenvalues of the posterior covariance.
    """            

    # Unpack the prior
    prior_mean = LR_prior_model['mean']
    prior_cov_inv = np.linalg.inv(LR_prior_model['cov'])  # Invert covariance
    
    # Number of state/action pairs:
    num_sa_pairs = observation_matrix.shape[1]

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via the Laplace approximation:    
    if r_init == []:
        r_init = 0.5 * np.ones(num_sa_pairs)    # Initial guess

    res = minimize(logreg_objective, r_init, args = (observation_matrix, 
                   preference_labels, prior_mean, prior_cov_inv),
                   method = 'L-BFGS-B', jac = logreg_obj_gradient)
    
    post_mean = res.x

    # Approximate inverse of the posterior covariance by evaluating the 
    # objective function's Hessian at the MAP estimate:
    post_cov_inverse = logreg_obj_hessian(post_mean, observation_matrix, 
                                  preference_labels, prior_cov_inv)

    # Calculate the eigenvectors and eigenvalues of the inverse posterior
    # covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov_inverse)

    # Invert the eigenvalues to get the eigenvalues corresponding to the
    # covariance matrix:
    evals = cov_scale / evals
    
    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals}



def logreg_objective(r, X, y, r0, prior_cov_inv):
    """
    Evaluate the optimization objective function for finding the posterior
    mean of the Bayesian logistic regression model via the Laplace 
    approximation; the posterior mean is the minimum of this (convex) objective 
    function.

    Inputs (note: d is the number of state/action pairs; n is the number of 
            data points/observations/rows in the observation matrix):
        1) r: the "point" at which to evaluate the objective function. This is
           a length-d vector.
        2) X: n-times-d observation matrix.
        3) y: length-n vector of labels (all values should be +/- 1)
        4) r0: prior value for r (length-d vector)
        5) prior_cov_inv: d x d inverse of the prior covariance matrix

    Output: the objective function evaluated at the given point (r).
    """
    
    obj = 0.5 * (r - r0) @ prior_cov_inv @ (r - r0)
    
    for i in range(len(y)):
        
        obj += np.log(1 + np.exp(-y[i] * np.dot(X[i, :], r)))
        
    return obj
            


def logreg_obj_gradient(r, X, y, r0, prior_cov_inv):
    """
    Evaluate the gradient of the Laplace approximation objective function.

    Inputs: same as in logreg_objective.

    Output: the objective function's gradient evaluated at the given point (r).
    """
    
    grad = prior_cov_inv @ (r - r0)
    
    for i in range(len(y)):
        
        yi = y[i]; xi = X[i, :]
        
        grad -= yi * xi / (1 + np.exp(yi * np.dot(xi, r)))
        
    return grad  
    

def logreg_obj_hessian(r, X, y, prior_cov_inv):
    """
    Evaluate the Hessian of the Laplace approximation objective function.

    Inputs: same as in logreg_GP_objective, except without r0.

    Output: the objective function's Hessian matrix evaluated at the given
            point (r).
    """
    
    hess = prior_cov_inv
    
    for i in range(len(y)):
        
        yi = y[i]; xi = X[i, :]
        xi_vec = xi.reshape((len(xi), 1))
        
        hess += (xi_vec @ np.transpose(xi_vec)) * np.exp(yi * np.dot(xi, r)) /   \
                (np.exp(yi * np.dot(xi, r)) + 1)**2
        
    return hess


