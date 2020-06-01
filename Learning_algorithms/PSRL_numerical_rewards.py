# -*- coding: utf-8 -*-
"""
Implementation of the posterior sampling RL algorithm (PSRL), as described in 
"(More) Efficient Reinforcement Learning via Posterior Sampling," by I. Osband,
B. Van Roy, and D. Russo (2013).

Unlike preference-based learning algorithms, PSRL receives numerical reward 
feedback after every step of interaction between the agent and the environment.
"""

import numpy as np
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../") 
from ValueIteration import value_iteration


def PSRL(time_horizon, NG_prior_params, env, num_iter, diri_prior = 1,
         run_str = '', seed = None):
    """
    This function implements the PSRL algorithm for performing posterior 
    sampling with numerical rewards at every step.
    
    Inputs:
        1) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.
        2) NG_prior_params: the hyperparameters for the normal-gamma model
           used for learning the posterior over rewards. This is a length-4 
           list of the form [mu0, k0, alpha0, beta0]. The normal-gamma
           prior is defined as NG(mu, lambda | mu0, k0, alpha0, beta0) = 
           Normal(mu | mu0, (k0 * lambda)^(-1)) * Gamma(lambda | alpha0, 
           rate = beta0). For details on the normal-gamma distribution, see 
           "Conjugate Bayesian analysis of the Gaussian distribution" by Kevin
           P. Murphy, https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
        3) env: the RL environment.
        4) num_iter: the number of iterations of the learning algorithm to run.
           Note that one trajectory rollout occurs per iteration of learning.
        5) diri_prior: parameter for setting the prior of the transition
           dynamics model. For each state/action pair, the Dirichlet prior is
           set to diri_prior * np.ones(num_states), where num_states is the 
           number of states in the MDP.
        6) run_str: if desired, a string with information about the current
           call to PSRL (e.g. hyperparameter values or repetition number), 
           which can be useful for print statements to track progress.
        7) seed: seed for random number generation.
 
    
    Returns: a vector of rewards received as the algorithm runs. This is either
             a) the total rewards from each trajectory rollout, or b) the 
             rewards at every step taken in the environment (the environment
             determines whether a) or b) is used).
    """
    
    if not seed is None:
        np.random.seed(seed)

    # Numbers of states and actions in the environment:
    num_states = env.nS
    num_actions = env.nA

    # Dirichlet model posterior over state/action transition probabilities.
    # Initially, this is set to the Dirichlet prior, and it's updated after
    # each observed state transition. Note that dirichlet_posterior[state][action] 
    # is a length-num_states array, specifying the probability distribution for
    # transitioning to each possible subsequent state from the given state/action. 
    # Setting diri_prior = 1 gives a uniform prior over transition probabilities.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: \
                                            diri_prior * np.ones(num_states)))
    
    # Initialize posterior parameters used for sampling from the reward model
    # (initially, these are equal to the prior parameters):
    NG_params = np.tile(NG_prior_params, (num_states, num_actions, 1))

    # Store how many times each state/action pair gets visited:
    visit_counts = np.zeros((num_states, num_actions))
    
    # Store rewards observed in each state/action:
    reward_samples = defaultdict(lambda: [])

    num_policies = 1     # Number of policies to sample per learning iteration

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
        print('PSRL, parameters %s: iteration = %i' % (run_str, iteration + 1))
 
        # Sample policies:
        policies, reward_models = advance(num_policies, dirichlet_posterior, 
                        NG_params, num_states, num_actions, time_horizon)
            
        for policy in policies:    # Roll out an action sequence
    
            state = env.reset()
            
            for t in range(time_horizon):
                
                action = np.random.choice(num_actions, p = policy[t, state, :])
                
                next_state, reward, done, = env.step(action)

                # Update state transition posterior:
                dirichlet_posterior[state][action][next_state] += 1
                
                # Update state/action visits counts:
                visit_counts[state][action] += 1
                
                # Store observed rewards:
                reward_samples[state, action].append(reward)
                
                # Tracking rewards for evaluation purposes (in case of 
                # tracking rewards at every single step):
                if not env.store_episode_reward:
                    rewards[reward_count] = env.get_step_reward(state, 
                                action, next_state)
                    reward_count += 1

                # Terminate trajectory if environment turns on "done" flag.
                if done:
                    break   

                state = next_state    

        # Tracking rewards for evaluation purposes (in case of tracking
        # rewards just over entire episodes):
        if env.store_episode_reward:

            rewards[reward_count] = env.get_episode_return()
            reward_count += 1
                
        # Call feedback function to update the normal-gamma reward posterior:
        NG_params = feedback_NG(NG_prior_params, visit_counts, reward_samples,
                             num_states, num_actions)
        
    # Return performance results:
    return rewards


def advance(num_policies, dirichlet_posterior, num_states, num_actions, 
            NG_params, time_horizon):
    """
    Draw a specified number of samples from the model posteriors over the 
    environment (i.e., the transition dynamics and rewards). For each sampled 
    environment, run value iteration to obtain the optimal policy given the
    sampled dynamics and rewards.
    
    This function assumes that the reward model posterior is an independent
    normal-gamma distribution for each state/action pair.

    Inputs: (note: d = num_states * num_actions, the number of state/action pairs)
        1) num_policies: the number of samples to draw from the posterior; a
           positive integer.
        2) dirichlet_posterior: the model posterior over transition dynamics
           parameters: dirichlet_posterior[state][action] is a length-num_states
           array of the Dirichlet parameters for the given state and action.
           These give the probability distribution of transitioning to each 
           possible subsequent state from the given state and action.
        3) num_states: number of states in the MDP.
        4) num_actions: number of actions in the MDP.
        5) NG_params: these parameters specify the normal-gamma reward posterior.
           It's a matrix of size num_states x num_actions x 4. NG_params[s, a, :]
           gives the 4 parameters of the normal-gamma model for state/action
           pair (s, a). This is a length-4 list of the form [mu_n, k_n, alpha_n, 
           beta_n]. The normal-gamma posterior is defined as:
               NG(mu, lambda | mu_n, k_n, alpha_n, beta_n) = 
                  Normal(mu | mu_n, (k_n * lambda)^(-1)) * Gamma(lambda | 
                  alpha_n, rate = beta_n).
                  
           For details on the normal-gamma distribution, see 
           "Conjugate Bayesian analysis of the Gaussian distribution" by Kevin
           P. Murphy, https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
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
    reward_models = []
    
    for i in range(num_policies):
        
        # Sample state transition dynamics from Dirichlet posterior:
        dynamics_sample = []
        
        for state in range(num_states):
            
            dynamics_sample_ = []
            
            for action in range(num_actions):
        
                dynamics_sample_.append(np.random.dirichlet(dirichlet_posterior[state][action]))
                
            dynamics_sample.append(dynamics_sample_)
    
        # Sample rewards from Normal-Gamma posterior:
        R = np.empty((num_states, num_actions))
        
        for s in range(num_states):
            for a in range(num_actions):
                
                gamma_sample = np.random.gamma(NG_params[s, a, 2], 1 / NG_params[s, a, 3])
                R[s, a] = np.random.normal(NG_params[s, a, 0], 
                         (NG_params[s, a, 1] * gamma_sample)**(-0.5))

        # Value iteration to determine policy:             
        policies.append(value_iteration(dynamics_sample, R, num_states, 
                                        num_actions, epsilon = 0,
                                        H = time_horizon)[0])
        reward_models.append(R)
        
    return policies, reward_models   


def feedback_NG(NG_prior_params, visit_counts, reward_samples, num_states, 
             num_actions):
    """
    This function updates the Normal-Gamma reward posterior based upon the 
    observed data.

    1) NG_prior_params: the hyperparameters for the normal-gamma model
       used for learning the posterior over rewards. This is a length-4 
       list of the form [mu0, k0, alpha0, beta0]. The normal-gamma
       prior is defined as NG(mu, lambda | mu0, k0, alpha0, beta0) = 
       Normal(mu | mu0, (k0 * lambda)^(-1)) * Gamma(lambda | alpha0, 
       rate = beta0). For details on the normal-gamma distribution, see 
       "Conjugate Bayesian analysis of the Gaussian distribution" by Kevin
       P. Murphy, https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
    2) visit_counts: num_states x num_actions matrix recording how many times
       each state/action pair has been visited.
    3) reward_samples: dictionary for which reward_samples[s][a] is a list of
       all the rewards observed on visits (so far) to state/action pair (s, a).
    4) num_states: number of states in the MDP.
    5) num_actions: number of actions in the MDP.

    Output: normal-gamma posterior. This is a matrix of size num_states x 
            num_actions x 4. NG_params[s, a, :] gives the 4 parameters of the 
            normal-gamma model for state/action pair (s, a). This is a length-4 
            list of the form [mu_n, k_n, alpha_n, beta_n]. The normal-gamma 
            posterior is defined as:
               NG(mu, lambda | mu_n, k_n, alpha_n, beta_n) = 
                  Normal(mu | mu_n, (k_n * lambda)^(-1)) * Gamma(lambda | 
                  alpha_n, rate = beta_n).

    """
    
    # To store the normal-gamma posterior:           
    NG_params = np.empty((num_states, num_actions, 4))
    
    mu0 = NG_prior_params[0]     # Unpack prior parameters
    k0 = NG_prior_params[1]
    alpha0 = NG_prior_params[2]
    beta0 = NG_prior_params[3]
    
    # Calculate posterior for each state/action pair:
    for s in range(num_states):
        for a in range(num_actions):
            
            n = visit_counts[s, a]
            
            if n == 0:
                NG_params[s, a] = NG_prior_params
                continue
            
            samples = np.array(reward_samples[s, a])
            avg = np.mean(samples)
            
            NG_params[s, a, 0] = (k0 * mu0 + n * avg) / (k0 + n)
            NG_params[s, a, 1] = k0 + n 
            NG_params[s, a, 2] = alpha0 + n/2
            NG_params[s, a, 3] = beta0 + 0.5 * np.sum((samples - avg)**2) + \
                                 k0 * n * (avg - mu0)**2 / (2 * (k0 + n))
    
    return NG_params
    
